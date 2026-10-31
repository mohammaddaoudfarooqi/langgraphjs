import {
  type AnyClientBulkWriteModel,
  type CreateIndexesOptions,
  type Document,
  type MongoClient,
  type Db as MongoDatabase,
} from "mongodb";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  BaseCheckpointSaver,
  type ChannelVersions,
  type Checkpoint,
  type CheckpointListOptions,
  type CheckpointTuple,
  type SerializerProtocol,
  type PendingWrite,
  type CheckpointMetadata,
  type DeltaChannelHistory,
  CheckpointPendingWrite,
  WRITES_IDX_MAP,
} from "@langchain/langgraph-checkpoint";
import {
  CheckpointIntegrityError,
  IndexMigrationRequired,
  ThreadExpiredError,
  type ChannelSize,
  type DocumentSizeInfo,
} from "./errors.js";
import {
  CompressingSerializer,
  MongoDBSerializer,
  type CompressionOptions,
} from "./serde.js";
import {
  buildBlobDocs,
  buildManifest,
  compareWrites,
  DEFAULT_INLINE_MAX_CHARS,
  isLayout2,
  manifestDocument,
  mergeBlobs,
  missingRefs,
  type BlobRef,
} from "./layout.js";
import { nextVersion, versionKind } from "./versions.js";
import {
  checkSize,
  DEFAULT_MAX_DOCUMENT_BYTES,
  DEFAULT_SIZE_WARNING_RATIO,
} from "./size.js";

export type MongoDBSaverParams = {
  client: MongoClient;
  dbName?: string;
  checkpointCollectionName?: string;
  checkpointWritesCollectionName?: string;
  /**
   * When true, writes an `upserted_at` BSON date to documents on every upsert.
   * Useful for MongoDB TTL indexes, auditing, or debugging.
   */
  enableTimestamps?: boolean;
  /**
   * Time-to-live in seconds for checkpoint documents. When set, an
   * `upserted_at` timestamp is written on every upsert (implies
   * `enableTimestamps`) and {@link MongoDBSaver.setup} creates MongoDB TTL
   * indexes so documents expire after the configured period of inactivity.
   */
  ttl?: number;
  /**
   * Largest BSON document the saver will send, in bytes. Larger documents
   * raise {@link CheckpointSizeError} before anything is written.
   * Defaults to 16 MiB minus 16 KiB.
   */
  maxDocumentBytes?: number;
  /**
   * Documents above `sizeWarningRatio × maxDocumentBytes` log a warning and
   * are reported to `onDocumentSize` with their largest channels.
   * Defaults to 0.8.
   */
  sizeWarningRatio?: number;
  /** Called with the collection and BSON size of every document written. */
  onDocumentSize?: (info: DocumentSizeInfo) => void;
  /**
   * Strings longer than this many characters are stored in
   * `checkpoint_blobs` rather than inline in the manifest. Defaults to 1024.
   */
  inlineMaxChars?: number;
  /**
   * Compress blob and pending-write values of at least `minBytes` with
   * `codec` (`"gzip"` by default; `"zstd"` where Node provides it). Off by
   * default. Compressed values are always readable, whether or not this is
   * set.
   */
  compression?: CompressionOptions;
  /** Ancestors read per query by `getDeltaChannelHistory`. Defaults to 100. */
  historyBatchSize?: number;
  /** Collection holding per-channel values (layout 2). */
  checkpointBlobsCollectionName?: string;
  /**
   * Thread-scoped retention. Off by default; when enabled, every write
   * records `threads.last_active` and {@link MongoDBSaver.sweepExpired}
   * deletes inactive threads.
   */
  retention?: RetentionOptions;
};

export interface RetentionOptions {
  /** Defaults to `true` when a `retention` object is given. */
  enabled?: boolean;
  /** How long a swept thread's tombstone is kept. Defaults to 7 days. */
  tombstoneSeconds?: number;
  /** Defaults to `"threads"`. */
  threadsCollectionName?: string;
}

let ttlDeprecationWarned = false;

/** @internal Test helper: re-arms the one-time `ttl` deprecation warning. */
export function _resetDeprecationWarnings() {
  ttlDeprecationWarned = false;
}

export const DEFAULT_TOMBSTONE_SECONDS = 7 * 24 * 60 * 60;

export interface IndexSpec {
  collection: string;
  keys: Record<string, 1 | -1>;
  unique: boolean;
  options: CreateIndexesOptions;
}

/** The indexes `setup()` ensures (REQ-E-020). Shared with the scripts. */
export function requiredIndexes(names: {
  checkpoints: string;
  writes: string;
  blobs: string;
  threads?: string;
  tombstoneSeconds?: number;
}): IndexSpec[] {
  const specs: IndexSpec[] = [
    {
      collection: names.checkpoints,
      keys: { thread_id: 1, checkpoint_ns: 1, checkpoint_id: -1 },
      unique: true,
      options: { name: "thread_ns_checkpoint_idx", unique: true },
    },
    {
      collection: names.writes,
      keys: {
        thread_id: 1,
        checkpoint_ns: 1,
        checkpoint_id: 1,
        task_id: 1,
        idx: 1,
      },
      unique: true,
      options: { name: "thread_ns_checkpoint_task_idx", unique: true },
    },
    {
      collection: names.blobs,
      keys: { thread_id: 1, checkpoint_ns: 1, channel: 1, version: 1 },
      unique: true,
      options: { name: "thread_ns_channel_version_idx", unique: true },
    },
  ];
  if (names.threads !== undefined) {
    specs.push(
      {
        collection: names.threads,
        keys: { thread_id: 1 },
        unique: true,
        options: { name: "thread_id_idx", unique: true },
      },
      {
        collection: names.threads,
        keys: { tombstoned_at: 1 },
        unique: false,
        options: {
          name: "tombstoned_at_ttl_idx",
          expireAfterSeconds:
            names.tombstoneSeconds ?? DEFAULT_TOMBSTONE_SECONDS,
        },
      },
      {
        collection: names.threads,
        keys: { last_active: 1 },
        unique: false,
        options: { name: "last_active_idx" },
      }
    );
  }
  return specs;
}

function sameKeys(a: Record<string, unknown>, b: Record<string, unknown>) {
  const ak = Object.entries(a);
  const bk = Object.entries(b);
  return (
    ak.length === bk.length &&
    ak.every(([k, v], i) => bk[i][0] === k && Number(bk[i][1]) === Number(v))
  );
}

function getStringConfigValue(
  name: string,
  value: unknown,
  { required = false }: { required?: boolean } = {}
): string | undefined {
  if (value === undefined) {
    if (required) {
      throw new Error(`Invalid configurable.${name}: expected a string`);
    }
    return undefined;
  }

  if (value === null || typeof value !== "string") {
    throw new Error(`Invalid configurable.${name}: expected a string`);
  }

  return value;
}

/**
 * A LangGraph checkpoint saver backed by a MongoDB database.
 *
 * NOTE: you need to call .setup() the first time you're using your checkpointer.
 *
 * @example
 * ```typescript
 * const checkpointer = new MongoDBSaver({ client });
 * await checkpointer.setup();
 * ```
 */
export class MongoDBSaver extends BaseCheckpointSaver {
  protected client: MongoClient;

  protected db: MongoDatabase;

  checkpointCollectionName = "checkpoints";

  checkpointWritesCollectionName = "checkpoint_writes";

  checkpointBlobsCollectionName = "checkpoint_blobs";

  /** Resolved retention settings; undefined when retention is off. */
  protected retention?: Required<RetentionOptions>;

  protected enableTimestamps: boolean;

  protected ttl?: number;

  /** The serializer supplied by the caller, or {@link MongoDBSerializer}. */
  protected baseSerde: SerializerProtocol;

  protected maxDocumentBytes: number;

  protected inlineMaxChars: number;

  protected historyBatchSize: number;

  protected sizeWarningRatio: number;

  protected onDocumentSize?: (info: DocumentSizeInfo) => void;

  private get timestampOp() {
    return this.enableTimestamps
      ? ({ $currentDate: { upserted_at: true } } as const)
      : {};
  }

  constructor(
    {
      client,
      dbName,
      checkpointCollectionName,
      checkpointWritesCollectionName,
      enableTimestamps,
      ttl,
      maxDocumentBytes,
      sizeWarningRatio,
      onDocumentSize,
      checkpointBlobsCollectionName,
      retention,
      inlineMaxChars,
      compression,
      historyBatchSize,
    }: MongoDBSaverParams,
    serde?: SerializerProtocol
  ) {
    super(serde ?? new MongoDBSerializer());
    this.baseSerde = this.serde;
    // Always able to read compressed values; compresses blobs and pending
    // writes only when `compression` is configured (manifest metadata never).
    this.serde = new CompressingSerializer(
      this.baseSerde,
      compression === undefined ? undefined : { codec: "gzip", ...compression }
    );
    this.client = client;
    this.client.appendMetadata({
      name: "langgraphjs_checkpoint_saver",
    });
    this.db = this.client.db(dbName);
    this.checkpointCollectionName =
      checkpointCollectionName ?? this.checkpointCollectionName;
    this.checkpointWritesCollectionName =
      checkpointWritesCollectionName ?? this.checkpointWritesCollectionName;
    this.ttl = ttl;
    if (ttl != null && !ttlDeprecationWarned) {
      ttlDeprecationWarned = true;
      console.warn(
        "[MongoDBSaver] The `ttl` option is deprecated: per-document TTL can " +
          "expire part of a thread (and does not expire per-channel blobs). " +
          "Use `retention` with `sweepExpired()` to delete whole inactive " +
          "threads instead."
      );
    }
    // TTL expiry relies on the `upserted_at` timestamp, so configuring a `ttl`
    // forces timestamps on (otherwise the TTL index would never match any
    // document and nothing would ever expire).
    this.enableTimestamps = (enableTimestamps ?? false) || ttl != null;
    this.maxDocumentBytes = maxDocumentBytes ?? DEFAULT_MAX_DOCUMENT_BYTES;
    this.sizeWarningRatio = sizeWarningRatio ?? DEFAULT_SIZE_WARNING_RATIO;
    this.onDocumentSize = onDocumentSize;
    this.inlineMaxChars = inlineMaxChars ?? DEFAULT_INLINE_MAX_CHARS;
    this.historyBatchSize = historyBatchSize ?? 100;
    this.checkpointBlobsCollectionName =
      checkpointBlobsCollectionName ?? this.checkpointBlobsCollectionName;
    if (retention !== undefined && (retention.enabled ?? true)) {
      this.retention = {
        enabled: true,
        tombstoneSeconds:
          retention.tombstoneSeconds ?? DEFAULT_TOMBSTONE_SECONDS,
        threadsCollectionName: retention.threadsCollectionName ?? "threads",
      };
    }
  }

  /** Measures a document about to be written and enforces the size limit. */
  protected async checkDocumentSize(
    collection: string,
    doc: object,
    channelSizes: () => ChannelSize[] | Promise<ChannelSize[]>
  ): Promise<number> {
    return checkSize(doc, {
      collection,
      limit: this.maxDocumentBytes,
      warningRatio: this.sizeWarningRatio,
      onDocumentSize: this.onDocumentSize,
      channelSizes,
    });
  }

  /** Serialized size of each channel value (only used for size reports). */
  protected async serializedChannelSizes(
    values: Record<string, unknown>
  ): Promise<ChannelSize[]> {
    return Promise.all(
      Object.entries(values).map(async ([channel, value]) => {
        const [, bytes] = await this.baseSerde.dumpsTyped(value);
        return { channel, bytes: bytes.length };
      })
    );
  }

  /**
   * Creates the indexes required by the checkpoint saver.
   *
   * Ensures **unique** compound indexes on `checkpoints`, `checkpoint_writes`
   * and `checkpoint_blobs` matching the query and upsert patterns, plus — when
   * retention is enabled — the `threads` indexes. When a `ttl` is configured,
   * additionally creates MongoDB TTL indexes on `upserted_at`.
   *
   * If a non-unique index with the same key pattern already exists (created by
   * an earlier version of this package), it is left untouched and an
   * {@link IndexMigrationRequired} error is returned for it; run
   * `scripts/check-unique-indexes.ts` to rebuild it.
   *
   * This method is idempotent and safe to call on every application start (and
   * concurrently). It returns an array of errors (empty if successful) so the
   * caller can decide how to handle failures.
   */
  async setup(): Promise<Error[]> {
    const operations: Promise<unknown>[] = requiredIndexes({
      checkpoints: this.checkpointCollectionName,
      writes: this.checkpointWritesCollectionName,
      blobs: this.checkpointBlobsCollectionName,
      threads: this.retention?.threadsCollectionName,
      tombstoneSeconds: this.retention?.tombstoneSeconds,
    }).map((spec) => this.ensureIndex(spec));

    if (this.ttl != null) {
      const ttlIndex = { upserted_at: 1 };
      const options = { expireAfterSeconds: this.ttl };
      operations.push(
        this.db
          .collection(this.checkpointCollectionName)
          .createIndex(ttlIndex, options),
        this.db
          .collection(this.checkpointWritesCollectionName)
          .createIndex(ttlIndex, options)
      );
    }

    const results = await Promise.allSettled(operations);
    return results
      .filter((r): r is PromiseRejectedResult => r.status === "rejected")
      .map((r) => r.reason as Error);
  }

  protected async ensureIndex(spec: IndexSpec): Promise<void> {
    const collection = this.db.collection(spec.collection);
    if (spec.unique) {
      const existing = await collection
        .listIndexes()
        .toArray()
        .catch((e) => {
          // NamespaceNotFound: the collection doesn't exist yet.
          if ((e as { code?: number }).code === 26) return [];
          throw e;
        });
      const same = existing.find((idx) => sameKeys(idx.key, spec.keys));
      if (same !== undefined) {
        if (same.unique === true) return;
        throw new IndexMigrationRequired({
          collection: spec.collection,
          indexName: same.name,
          keys: spec.keys,
        });
      }
    }
    await collection.createIndex(spec.keys, spec.options);
  }

  /** `maxWireVersion` / topology from one `hello`, cached per saver. */
  private serverInfoPromise?: Promise<ServerInfo>;

  protected async serverInfo(): Promise<ServerInfo> {
    if (this.serverInfoPromise === undefined) {
      this.serverInfoPromise = this.client
        .db("admin")
        .command({ hello: 1 })
        .then((hello) => ({
          maxWireVersion: Number(hello.maxWireVersion ?? 0),
          supportsTransactions:
            hello.setName !== undefined || hello.msg === "isdbgrid",
        }))
        .catch((e) => {
          this.serverInfoPromise = undefined;
          throw e;
        });
    }
    return this.serverInfoPromise;
  }

  /**
   * Next channel version (REQ-E-037/038): integer inputs (threads written by
   * releases <= 1.4.x) keep integer versions; `undefined` and strings produce
   * `"{n:032}.{r:016}"` strings, which select the per-channel layout.
   */
  getNextVersion(current: number | undefined): number {
    // The base class is typed for numbers; string versions are supported by
    // LangGraph core (compareChannelVersions) and returned here at runtime.
    return nextVersion(current as number | string | undefined) as number;
  }

  /**
   * Whether `checkpoint` is stored in the per-channel layout (layout 2).
   * Threads whose channel versions are all numbers keep the single-document
   * format (REQ-E-039, INV-002).
   */
  protected useLayout2(checkpoint: Checkpoint): boolean {
    return versionKind(checkpoint.channel_versions) !== "number";
  }

  /**
   * Retrieves a checkpoint from the MongoDB database based on the
   * provided config. If the config contains a "checkpoint_id" key, the checkpoint with
   * the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
   * for the given thread ID is retrieved.
   */
  async getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined> {
    const thread_id = getStringConfigValue(
      "thread_id",
      config.configurable?.thread_id
    );
    if (thread_id === undefined) {
      return undefined;
    }
    const checkpoint_ns =
      getStringConfigValue(
        "checkpoint_ns",
        config.configurable?.checkpoint_ns
      ) ?? "";
    const checkpoint_id = getStringConfigValue(
      "checkpoint_id",
      config.configurable?.checkpoint_id
    );

    let query;
    if (checkpoint_id) {
      query = {
        thread_id,
        checkpoint_ns,
        checkpoint_id,
      };
    } else {
      query = { thread_id, checkpoint_ns };
    }
    const result = await this.db
      .collection(this.checkpointCollectionName)
      .find(query)
      .sort("checkpoint_id", -1)
      .limit(1)
      .toArray();
    if (result.length === 0) {
      return undefined;
    }
    const doc = result[0];
    const configurableValues = {
      thread_id,
      checkpoint_ns,
      checkpoint_id: doc.checkpoint_id,
    };
    const [checkpoint, pendingWrites, metadata] = await Promise.all([
      this.loadCheckpoint(doc, thread_id, checkpoint_ns),
      this.loadPendingWrites(configurableValues),
      this.serde.loadsTyped(
        doc.type,
        binaryValue(doc.metadata)
      ) as Promise<CheckpointMetadata>,
    ]);
    return {
      config: { configurable: configurableValues },
      checkpoint,
      pendingWrites,
      metadata,
      parentConfig:
        doc.parent_checkpoint_id != null
          ? {
              configurable: {
                thread_id,
                checkpoint_ns,
                checkpoint_id: doc.parent_checkpoint_id,
              },
            }
          : undefined,
    };
  }

  /**
   * Deserializes the checkpoint stored in `doc`. Old-format documents are
   * returned exactly as today (INV-001); layout 2 manifests are merged with
   * the blobs listed in `blob_refs`, fetched with one query (REQ-E-035).
   */
  protected async loadCheckpoint(
    doc: Document,
    thread_id: string,
    checkpoint_ns: string
  ): Promise<Checkpoint> {
    if (!isLayout2(doc)) {
      return (await this.serde.loadsTyped(
        doc.type,
        binaryValue(doc.checkpoint)
      )) as Checkpoint;
    }
    const refs: BlobRef[] = (doc.blob_refs ?? []) as BlobRef[];
    const [manifestCheckpoint, blobs] = await Promise.all([
      this.serde.loadsTyped(
        doc.type,
        binaryValue(doc.checkpoint)
      ) as Promise<Checkpoint>,
      this.loadBlobs(thread_id, checkpoint_ns, doc.checkpoint_id, refs),
    ]);
    return mergeBlobs(manifestCheckpoint, blobs);
  }

  /** Fetches and deserializes `refs`; raises if any is missing (REQ-E-036). */
  protected async loadBlobs(
    thread_id: string,
    checkpoint_ns: string,
    checkpoint_id: string,
    refs: BlobRef[]
  ): Promise<{ channel: string; version: unknown; value: unknown }[]> {
    if (refs.length === 0) return [];
    const docs = await this.db
      .collection(this.checkpointBlobsCollectionName)
      .find({
        thread_id,
        checkpoint_ns,
        $or: refs.map((r) => ({ channel: r.channel, version: r.version })),
      })
      .toArray();
    const missing = missingRefs(refs, docs as unknown as BlobRef[]);
    if (missing.length > 0) {
      throw new CheckpointIntegrityError({
        threadId: thread_id,
        checkpointNs: checkpoint_ns,
        checkpointId: checkpoint_id,
        missing,
      });
    }
    return Promise.all(
      docs.map(async (b) => ({
        channel: b.channel as string,
        version: b.version,
        value: await this.serde.loadsTyped(b.type, binaryValue(b.value)),
      }))
    );
  }

  /**
   * Pending writes of one checkpoint, ordered by `(task_path, task_id, idx)`
   * (documents written before `task_path` was stored sort first).
   */
  protected async loadPendingWrites(ids: {
    thread_id: string;
    checkpoint_ns: string;
    checkpoint_id: string;
  }): Promise<CheckpointPendingWrite[]> {
    const serializedWrites = await this.db
      .collection(this.checkpointWritesCollectionName)
      .find(ids)
      .toArray();
    serializedWrites.sort((a, b) =>
      compareWrites(a as unknown as WriteKey, b as unknown as WriteKey)
    );
    return Promise.all(
      serializedWrites.map(
        async (serializedWrite) =>
          [
            serializedWrite.task_id,
            serializedWrite.channel,
            await this.serde.loadsTyped(
              serializedWrite.type,
              binaryValue(serializedWrite.value)
            ),
          ] as CheckpointPendingWrite
      )
    );
  }

  /**
   * Retrieve a list of checkpoint tuples from the MongoDB database based
   * on the provided config. The checkpoints are ordered by checkpoint ID
   * in descending order (newest first).
   */
  async *list(
    config: RunnableConfig,
    options?: CheckpointListOptions
  ): AsyncGenerator<CheckpointTuple> {
    const { limit, before, filter } = options ?? {};
    const query: Record<string, unknown> = {};
    const thread_id = getStringConfigValue(
      "thread_id",
      config.configurable?.thread_id
    );
    const checkpoint_ns = getStringConfigValue(
      "checkpoint_ns",
      config.configurable?.checkpoint_ns
    );

    if (thread_id) {
      query.thread_id = thread_id;
    }
    if (checkpoint_ns !== undefined) {
      query.checkpoint_ns = checkpoint_ns;
    }

    if (filter) {
      Object.entries(filter).forEach(([key, value]) => {
        // Prevent MongoDB operator injection - only allow primitive values
        if (value !== null && typeof value === "object") {
          throw new Error(
            `Invalid filter value for key "${key}": filter values must be primitives (string, number, boolean, or null)`
          );
        }
        query[`metadata_search.${key}`] = value;
      });
    }

    if (before?.configurable?.checkpoint_id !== undefined) {
      const before_checkpoint_id = getStringConfigValue(
        "checkpoint_id",
        before.configurable?.checkpoint_id
      );
      query.checkpoint_id = { $lt: before_checkpoint_id };
    }

    let result = this.db
      .collection(this.checkpointCollectionName)
      .find(query)
      .sort("checkpoint_id", -1);

    if (limit !== undefined) {
      result = result.limit(limit);
    }

    for await (const doc of result) {
      // Query pending writes for this checkpoint, matching getTuple() behavior
      const [checkpoint, metadata, pendingWrites] = await Promise.all([
        this.loadCheckpoint(doc, doc.thread_id, doc.checkpoint_ns),
        this.serde.loadsTyped(
          doc.type,
          binaryValue(doc.metadata)
        ) as Promise<CheckpointMetadata>,
        this.loadPendingWrites({
          thread_id: doc.thread_id,
          checkpoint_ns: doc.checkpoint_ns,
          checkpoint_id: doc.checkpoint_id,
        }),
      ]);

      yield {
        config: {
          configurable: {
            thread_id: doc.thread_id,
            checkpoint_ns: doc.checkpoint_ns,
            checkpoint_id: doc.checkpoint_id,
          },
        },
        checkpoint,
        pendingWrites,
        metadata,
        parentConfig: doc.parent_checkpoint_id
          ? {
              configurable: {
                thread_id: doc.thread_id,
                checkpoint_ns: doc.checkpoint_ns,
                checkpoint_id: doc.parent_checkpoint_id,
              },
            }
          : undefined,
      };
    }
  }

  /**
   * Saves a checkpoint to the MongoDB database. The checkpoint is associated
   * with the provided config and its parent config (if any).
   *
   * Threads with integer channel versions (written by releases <= 1.4.x) are
   * stored as one document, exactly as before. All other threads use the
   * per-channel layout: changed non-scalar channels (per `newVersions`) are
   * written to `checkpoint_blobs`, then a small manifest to `checkpoints`.
   */
  async put(
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    newVersions?: ChannelVersions
  ): Promise<RunnableConfig> {
    const thread_id = getStringConfigValue(
      "thread_id",
      config.configurable?.thread_id,
      { required: true }
    ) as string;
    const checkpoint_ns =
      getStringConfigValue(
        "checkpoint_ns",
        config.configurable?.checkpoint_ns
      ) ?? "";
    const parent_checkpoint_id = getStringConfigValue(
      "checkpoint_id",
      config.configurable?.checkpoint_id
    );
    const checkpoint_id = checkpoint.id;
    const upsertQuery = {
      thread_id,
      checkpoint_ns,
      checkpoint_id,
    };
    const timestamp = this.enableTimestamps ? { upserted_at: new Date() } : {};

    const blobOps: WriteOp[] = [];
    let manifestSet: Document;
    let channelSizes: () => Promise<ChannelSize[]>;

    if (this.useLayout2(checkpoint)) {
      const plan = buildManifest(checkpoint, {
        inlineMaxChars: this.inlineMaxChars,
      });
      const drafts = buildBlobDocs(plan, {
        threadId: thread_id,
        checkpointNs: checkpoint_ns,
        newVersions,
      });
      // Serialize and size-check every blob before writing anything.
      const blobDocs = await Promise.all(
        drafts.map(async ({ value, ...key }) => {
          const [type, bytes] = await this.serde.dumpsTyped(value);
          const doc = { ...key, type, value: bytes, ...timestamp };
          await this.checkDocumentSize(
            this.checkpointBlobsCollectionName,
            doc,
            () => [{ channel: key.channel, bytes: bytes.length }]
          );
          return doc;
        })
      );
      for (const {
        thread_id: t,
        checkpoint_ns: ns,
        channel,
        version,
        ...rest
      } of blobDocs) {
        blobOps.push({
          filter: { thread_id: t, checkpoint_ns: ns, channel, version },
          // Insert-if-absent: a blob is immutable once written (REQ-E-034).
          update: { $setOnInsert: rest },
        });
      }
      const [
        [checkpointType, serializedCheckpoint],
        [metadataType, serializedMetadata],
      ] = await Promise.all([
        this.baseSerde.dumpsTyped(plan.checkpoint),
        this.baseSerde.dumpsTyped(metadata),
      ]);
      if (checkpointType !== metadataType) {
        throw new Error("Mismatched checkpoint and metadata types.");
      }
      manifestSet = manifestDocument(plan, {
        parentCheckpointId: parent_checkpoint_id,
        type: checkpointType,
        checkpoint: serializedCheckpoint,
        metadata: serializedMetadata,
        metadataSearch: metadata,
      });
      channelSizes = () =>
        this.serializedChannelSizes(plan.checkpoint.channel_values);
    } else {
      const [
        [checkpointType, serializedCheckpoint],
        [metadataType, serializedMetadata],
      ] = await Promise.all([
        this.baseSerde.dumpsTyped(checkpoint),
        this.baseSerde.dumpsTyped(metadata),
      ]);
      if (checkpointType !== metadataType) {
        throw new Error("Mismatched checkpoint and metadata types.");
      }
      manifestSet = {
        parent_checkpoint_id,
        type: checkpointType,
        checkpoint: serializedCheckpoint,
        metadata: serializedMetadata,
        metadata_search: metadata,
      };
      channelSizes = () =>
        this.serializedChannelSizes(checkpoint.channel_values);
    }

    await this.checkDocumentSize(
      this.checkpointCollectionName,
      { ...upsertQuery, ...manifestSet, ...timestamp },
      channelSizes
    );

    await this.writeInOrder(thread_id, {
      first:
        blobOps.length > 0
          ? [{ collection: this.checkpointBlobsCollectionName, ops: blobOps }]
          : [],
      last: {
        collection: this.checkpointCollectionName,
        ops: [
          {
            filter: upsertQuery,
            update: { $set: manifestSet, ...this.timestampOp },
          },
        ],
      },
    });

    return {
      configurable: {
        thread_id,
        checkpoint_ns,
        checkpoint_id,
      },
    };
  }

  /** The `threads` upsert that records activity (REQ-E-060/062). */
  protected threadActivityOp(thread_id: string): WriteOp {
    return {
      // Never matches a tombstone: the upsert then collides with the
      // tombstone's unique `thread_id` and fails with a duplicate key error.
      filter: { thread_id, tombstone: { $exists: false } },
      update: {
        $max: { last_active: new Date() },
        $setOnInsert: { generation: 0 },
      },
    };
  }

  /**
   * Performs the writes of one `put`/`putWrites` with the round-trip budget of
   * REQ-NF-001 / REQ-E-041:
   *
   * - MongoDB 8.0+ (`maxWireVersion >= 25`), when there is more than one
   *   stage: one ordered client-level `bulkWrite` — `threads` upsert, then the
   *   `first` stage, then `last`;
   * - otherwise: the `threads` upsert, then the `first` stage (each an
   *   ordered collection `bulkWrite`), then `last`.
   *
   * All operations are upserts. A duplicate key error on the `threads` upsert
   * means the thread is tombstoned: {@link ThreadExpiredError}, and `last` is
   * never written.
   */
  protected async writeInOrder(
    thread_id: string,
    stages: {
      first: { collection: string; ops: WriteOp[] }[];
      last: { collection: string; ops: WriteOp[] };
    },
    retried = false
  ): Promise<void> {
    const threadsOp = this.retention
      ? this.threadActivityOp(thread_id)
      : undefined;
    const sequentialStages =
      (stages.first.length > 0 || threadsOp !== undefined ? 1 : 0) + 1;
    try {
      if (
        sequentialStages > 1 &&
        (await this.serverInfo()).maxWireVersion >= MIN_CLIENT_BULK_WRITE_WIRE
      ) {
        await this.clientBulkWrite(threadsOp, stages);
        return;
      }
      // The threads upsert must succeed before anything else is written, or
      // a tombstoned thread would keep values that no sweep ever removes.
      if (threadsOp !== undefined) {
        await this.collectionWrite(this.retention!.threadsCollectionName, [
          threadsOp,
        ]).catch((e) => {
          throw new ThreadsWriteError(e);
        });
      }
      await Promise.all(
        stages.first.map(({ collection, ops }) =>
          this.collectionWrite(collection, ops)
        )
      );
      await this.collectionWrite(stages.last.collection, stages.last.ops);
    } catch (e) {
      const cause = isThreadsWriteError(e) ? e.cause : e;
      if (threadsOp !== undefined && isThreadsDuplicateKey(e)) {
        // A duplicate key also occurs when two writers create the same new
        // thread concurrently; only a tombstone means the thread expired.
        const record = await this.db
          .collection(this.retention!.threadsCollectionName)
          .findOne({ thread_id });
        if (record?.tombstone === true) {
          throw new ThreadExpiredError(thread_id, { cause });
        }
        if (!retried) {
          await this.writeInOrder(thread_id, stages, true);
          return;
        }
      }
      throw cause;
    }
  }

  private async collectionWrite(collection: string, ops: WriteOp[]) {
    const coll = this.db.collection(collection);
    if (ops.length === 1) {
      await coll.updateOne(ops[0].filter, ops[0].update, { upsert: true });
    } else {
      await coll.bulkWrite(
        ops.map((op) => ({
          updateOne: { filter: op.filter, update: op.update, upsert: true },
        })),
        { ordered: true }
      );
    }
  }

  private async clientBulkWrite(
    threadsOp: WriteOp | undefined,
    stages: {
      first: { collection: string; ops: WriteOp[] }[];
      last: { collection: string; ops: WriteOp[] };
    }
  ) {
    const dbName = this.db.databaseName;
    const models: AnyClientBulkWriteModel<Document>[] = [];
    const add = (collection: string, op: WriteOp) =>
      models.push({
        namespace: `${dbName}.${collection}`,
        name: "updateOne",
        filter: op.filter,
        update: op.update,
        upsert: true,
      });
    if (threadsOp !== undefined) {
      add(this.retention!.threadsCollectionName, threadsOp);
    }
    for (const { collection, ops } of stages.first) {
      for (const op of ops) add(collection, op);
    }
    for (const op of stages.last.ops) add(stages.last.collection, op);
    try {
      await this.client.bulkWrite(models, { ordered: true });
    } catch (e) {
      const writeErrors = (
        e as { writeErrors?: Map<number, { code?: number }> }
      ).writeErrors;
      if (threadsOp !== undefined && writeErrors?.get?.(0)?.code === 11000) {
        throw new ThreadsWriteError(e);
      }
      throw e;
    }
  }

  /**
   * Saves intermediate writes associated with a checkpoint to the MongoDB database.
   */
  async putWrites(
    config: RunnableConfig,
    writes: PendingWrite[],
    taskId: string,
    taskPath?: string
  ): Promise<void> {
    const thread_id = getStringConfigValue(
      "thread_id",
      config.configurable?.thread_id,
      { required: true }
    ) as string;
    const checkpoint_ns = getStringConfigValue(
      "checkpoint_ns",
      config.configurable?.checkpoint_ns,
      { required: true }
    );
    const checkpoint_id = getStringConfigValue(
      "checkpoint_id",
      config.configurable?.checkpoint_id,
      { required: true }
    );

    // Conflict resolution matches the Python MongoDB checkpointer and the
    // langgraph-checkpoint contract (BaseCheckpointSaver.put_writes):
    //   - When every write targets a special channel (ERROR / SCHEDULED /
    //     INTERRUPT / RESUME, each pinned to a negative `idx` by
    //     WRITES_IDX_MAP), we $set so e.g. INTERRUPT can be overwritten on
    //     RESUME.
    //   - Otherwise we $setOnInsert so a regular write from one task can
    //     never clobber a regular write that another concurrent task already
    //     stored at the same (task_id, idx).
    const allSpecial = writes.every(([channel]) => channel in WRITES_IDX_MAP);

    const operations: WriteOp[] = await Promise.all(
      writes.map(async ([channel, value], idx) => {
        const upsertQuery = {
          thread_id,
          checkpoint_ns,
          checkpoint_id,
          task_id: taskId,
          // Special channels are stored at fixed negative indices so they
          // never collide with regular per-step writes (whose `idx` is the
          // ordinal within `writes`).
          idx: WRITES_IDX_MAP[channel] ?? idx,
        };

        const [type, serializedValue] = await this.serde.dumpsTyped(value);
        // `task_path` is the serialized task path; `""` sorts first (see
        // `writesSortKey`). It is not part of the upsert filter, so rows
        // written before it existed keep matching.
        const fields = {
          channel,
          type,
          value: serializedValue,
          task_path: taskPath ?? "",
        };
        await this.checkDocumentSize(
          this.checkpointWritesCollectionName,
          {
            ...upsertQuery,
            ...fields,
            ...(this.enableTimestamps ? { upserted_at: new Date() } : {}),
          },
          () => [{ channel, bytes: serializedValue.length }]
        );

        return {
          filter: upsertQuery,
          update: allSpecial
            ? { $set: fields, ...this.timestampOp }
            : {
                // Insert-or-ignore: the row's channel/value AND its
                // upserted_at timestamp must only be written on first insert
                // so a no-op update against a peer task's existing row
                // doesn't bump that row's "last modified" stamp.
                $setOnInsert: this.enableTimestamps
                  ? { ...fields, upserted_at: new Date() }
                  : fields,
              },
        };
      })
    );

    // The MongoDB driver rejects `bulkWrite([])` with "Invalid BulkOperation,
    // Batch cannot be empty". `writes` can be empty in human-in-the-loop /
    // `interrupt()` flows, so skip the call when there is nothing to persist
    // (INV-004; `threads` isn't touched either).
    if (operations.length === 0) return;

    if (this.retention === undefined) {
      await this.db.collection(this.checkpointWritesCollectionName).bulkWrite(
        operations.map((op) => ({
          updateOne: { filter: op.filter, update: op.update, upsert: true },
        }))
      );
      return;
    }
    await this.writeInOrder(thread_id, {
      first: [],
      last: {
        collection: this.checkpointWritesCollectionName,
        ops: operations,
      },
    });
  }

  /**
   * Reconstructs `DeltaChannel` history with range scans instead of one
   * `getTuple` per ancestor (REQ-E-080…083). Returns exactly what the base
   * implementation returns for the same data, except that a broken
   * `parent_checkpoint_id` chain raises {@link CheckpointIntegrityError}
   * instead of silently truncating.
   *
   * Ancestors are read `historyBatchSize` at a time, newest first, with a
   * projection of only the identity and layout fields. Checkpoint IDs are
   * time-ordered (uuid6), so every ancestor sorts before its child; fork
   * siblings in a batch are skipped. Pending writes are fetched with one
   * `$in` query per batch of on-path ancestors and ordered by
   * `(task_path, task_id, idx)`. Only seeds are deserialized from payloads
   * (old-format documents must be read whole to locate their channels).
   */
  async getDeltaChannelHistory(options: {
    config: RunnableConfig;
    channels: string[];
  }): Promise<Record<string, DeltaChannelHistory>> {
    const { config, channels } = options;
    if (channels.length === 0) return {};
    const empty = () =>
      Object.fromEntries(
        channels.map((ch) => [ch, { writes: [] } as DeltaChannelHistory])
      );
    const thread_id = getStringConfigValue(
      "thread_id",
      config.configurable?.thread_id
    );
    if (thread_id === undefined) return empty();
    const checkpoint_ns =
      getStringConfigValue(
        "checkpoint_ns",
        config.configurable?.checkpoint_ns
      ) ?? "";
    const checkpoint_id = getStringConfigValue(
      "checkpoint_id",
      config.configurable?.checkpoint_id
    );

    const checkpoints = this.db.collection(this.checkpointCollectionName);
    const batchSize = this.historyBatchSize;
    const projection = {
      _id: 0,
      checkpoint_id: 1,
      parent_checkpoint_id: 1,
      layout: 1,
      blob_refs: 1,
      inline_channels: 1,
    };
    const scan = (upTo?: string) =>
      checkpoints
        .find(
          upTo === undefined
            ? { thread_id, checkpoint_ns }
            : { thread_id, checkpoint_ns, checkpoint_id: { $lte: upTo } },
          { projection }
        )
        .sort({ checkpoint_id: -1 })
        .limit(batchSize)
        .toArray();

    // The first scan also resolves the target.
    let batch = await scan(checkpoint_id || undefined);
    const target = batch[0];
    if (
      target === undefined ||
      (checkpoint_id && target.checkpoint_id !== checkpoint_id)
    ) {
      return empty();
    }
    let byId = new Map(batch.map((d) => [d.checkpoint_id as string, d]));

    const remaining = new Set(channels);
    // On-path ancestors (excluding the target) with the channels whose
    // writes they contribute.
    const path: { checkpointId: string; channels: string[] }[] = [];
    type Seed =
      | { kind: "blob"; ref: BlobRef }
      | { kind: "manifest"; checkpointId: string };
    const seeds = new Map<string, Seed>();
    const oldPayloads = new Map<string, Promise<Checkpoint>>();
    const loadOldPayloads = (docs: Document[]) => {
      const ids = docs
        .filter((d) => !isLayout2(d) && !oldPayloads.has(d.checkpoint_id))
        .map((d) => d.checkpoint_id as string);
      if (ids.length === 0) return;
      const fetched = checkpoints
        .find(
          { thread_id, checkpoint_ns, checkpoint_id: { $in: ids } },
          { projection: { _id: 0, checkpoint_id: 1, type: 1, checkpoint: 1 } }
        )
        .toArray();
      for (const id of ids) {
        oldPayloads.set(
          id,
          fetched.then(async (rows) => {
            const row = rows.find((r) => r.checkpoint_id === id)!;
            return (await this.serde.loadsTyped(
              row.type,
              binaryValue(row.checkpoint)
            )) as Checkpoint;
          })
        );
      }
    };

    let child = target;
    let cursor: string | undefined = target.parent_checkpoint_id ?? undefined;
    while (cursor != null && remaining.size > 0) {
      let doc = byId.get(cursor);
      if (doc === undefined) {
        const minId = batch[batch.length - 1]?.checkpoint_id as string;
        if (batch.length === batchSize && cursor < minId) {
          batch = await scan(cursor);
          byId = new Map(batch.map((d) => [d.checkpoint_id as string, d]));
          doc = byId.get(cursor);
        }
        if (doc === undefined) {
          throw new CheckpointIntegrityError({
            threadId: thread_id,
            checkpointNs: checkpoint_ns,
            checkpointId: child.checkpoint_id,
            missing: [{ checkpoint_id: cursor }],
            message:
              `Ancestor ${cursor} of checkpoint ${child.checkpoint_id} ` +
              `(thread "${thread_id}", namespace "${checkpoint_ns}") is missing`,
          });
        }
      }
      path.push({ checkpointId: cursor, channels: [...remaining] });
      if (isLayout2(doc)) {
        const inline = new Set<string>(doc.inline_channels ?? []);
        const refs = new Map<string, BlobRef>(
          ((doc.blob_refs ?? []) as BlobRef[]).map((r) => [r.channel, r])
        );
        for (const ch of [...remaining]) {
          if (inline.has(ch)) {
            seeds.set(ch, { kind: "manifest", checkpointId: cursor });
            remaining.delete(ch);
          } else if (refs.has(ch)) {
            seeds.set(ch, { kind: "blob", ref: refs.get(ch)! });
            remaining.delete(ch);
          }
        }
      } else {
        // Old format: the channel list is only in the payload.
        loadOldPayloads(batch);
        const payload = await oldPayloads.get(cursor)!;
        for (const ch of [...remaining]) {
          if (
            Object.prototype.hasOwnProperty.call(payload.channel_values, ch)
          ) {
            seeds.set(ch, { kind: "manifest", checkpointId: cursor });
            remaining.delete(ch);
          }
        }
      }
      child = doc;
      cursor = doc.parent_checkpoint_id ?? undefined;
    }

    // Seeds and writes, in parallel.
    const blobSeeds = [...seeds.entries()].filter(
      (e): e is [string, { kind: "blob"; ref: BlobRef }] => e[1].kind === "blob"
    );
    const manifestSeedIds = [
      ...new Set(
        [...seeds.values()]
          .filter(
            (s): s is { kind: "manifest"; checkpointId: string } =>
              s.kind === "manifest"
          )
          .map((s) => s.checkpointId)
      ),
    ];
    const blobSeedValues = this.loadBlobs(
      thread_id,
      checkpoint_ns,
      target.checkpoint_id,
      blobSeeds.map(([, s]) => s.ref)
    );
    const manifestPayloads = (async () => {
      const out = new Map<string, Checkpoint>();
      const missingIds = manifestSeedIds.filter((id) => !oldPayloads.has(id));
      if (missingIds.length > 0) {
        const rows = await checkpoints
          .find(
            { thread_id, checkpoint_ns, checkpoint_id: { $in: missingIds } },
            { projection: { _id: 0, checkpoint_id: 1, type: 1, checkpoint: 1 } }
          )
          .toArray();
        for (const row of rows) {
          out.set(
            row.checkpoint_id,
            (await this.serde.loadsTyped(
              row.type,
              binaryValue(row.checkpoint)
            )) as Checkpoint
          );
        }
      }
      for (const id of manifestSeedIds) {
        if (oldPayloads.has(id)) out.set(id, await oldPayloads.get(id)!);
      }
      return out;
    })();
    const writesColl = this.db.collection(this.checkpointWritesCollectionName);
    const writeBatches: Promise<Document[]>[] = [];
    for (let i = 0; i < path.length; i += batchSize) {
      const ids = path.slice(i, i + batchSize).map((p) => p.checkpointId);
      writeBatches.push(
        writesColl
          .find({
            thread_id,
            checkpoint_ns,
            checkpoint_id: { $in: ids },
            channel: { $in: channels },
          })
          .toArray()
      );
    }
    const [blobValues, payloads, writeDocs] = await Promise.all([
      blobSeedValues,
      manifestPayloads,
      Promise.all(writeBatches).then((b) => b.flat()),
    ]);

    const writesByCheckpoint = new Map<string, Document[]>();
    for (const w of writeDocs) {
      const list = writesByCheckpoint.get(w.checkpoint_id) ?? [];
      list.push(w);
      writesByCheckpoint.set(w.checkpoint_id, list);
    }
    const collected: Record<string, Document[]> = {};
    for (const ch of channels) collected[ch] = [];
    // Oldest ancestor first; each checkpoint's writes in canonical order.
    for (let i = path.length - 1; i >= 0; i -= 1) {
      const { checkpointId, channels: active } = path[i];
      const docs = (writesByCheckpoint.get(checkpointId) ?? []).sort((a, b) =>
        compareWrites(a as unknown as WriteKey, b as unknown as WriteKey)
      );
      const activeSet = new Set(active);
      for (const w of docs) {
        if (activeSet.has(w.channel)) collected[w.channel].push(w);
      }
    }

    const result: Record<string, DeltaChannelHistory> = {};
    for (const ch of channels) {
      const writes = await Promise.all(
        collected[ch].map(
          async (w) =>
            [
              w.task_id,
              w.channel,
              await this.serde.loadsTyped(w.type, binaryValue(w.value)),
            ] as CheckpointPendingWrite
        )
      );
      const entry: DeltaChannelHistory = { writes };
      const seed = seeds.get(ch);
      if (seed?.kind === "blob") {
        entry.seed = blobValues.find((b) => b.channel === ch)!.value;
      } else if (seed?.kind === "manifest") {
        entry.seed = payloads.get(seed.checkpointId)!.channel_values[ch];
      }
      result[ch] = entry;
    }
    return result;
  }

  /**
   * Deletes threads inactive since before `olderThan` (REQ-E-061…063).
   * Requires `retention`; the application schedules the calls.
   *
   * Each expired thread is first claimed with a conditional update that sets
   * `tombstone` (so concurrent writes fail with {@link ThreadExpiredError}),
   * then its manifests, pending writes and blobs are deleted. The `threads`
   * record stays as a tombstone until the TTL index removes it after
   * `retention.tombstoneSeconds`. Threads claimed by an interrupted earlier
   * sweep are finished first.
   *
   * @returns the number of threads swept and of documents deleted.
   */
  async sweepExpired(options: {
    olderThan: Date;
    batchSize?: number;
  }): Promise<{ threads: number; documents: number }> {
    if (this.retention === undefined) {
      throw new Error("sweepExpired requires the `retention` option");
    }
    const batchSize = options.batchSize ?? 100;
    const threads = this.db.collection(this.retention.threadsCollectionName);
    let swept = 0;
    let documents = 0;

    // Claimed but not yet purged (purging unsets `last_active`).
    const pending = await threads
      .find(
        { tombstone: true, last_active: { $exists: true } },
        { projection: { thread_id: 1 } }
      )
      .limit(batchSize)
      .toArray();
    for (const { thread_id } of pending) {
      documents += await this.purgeThread(thread_id);
      swept += 1;
    }

    while (swept < batchSize) {
      const claimed = await threads.findOneAndUpdate(
        {
          last_active: { $lt: options.olderThan },
          tombstone: { $exists: false },
        },
        {
          $set: { tombstone: true, tombstoned_at: new Date() },
          $inc: { generation: 1 },
        },
        { projection: { thread_id: 1 }, returnDocument: "after" }
      );
      if (claimed === null) break;
      documents += await this.purgeThread(claimed.thread_id);
      swept += 1;
    }
    return { threads: swept, documents };
  }

  /** Deletes a claimed thread's data and marks its tombstone as purged. */
  protected async purgeThread(thread_id: string): Promise<number> {
    let documents = 0;
    for (const name of [
      this.checkpointCollectionName,
      this.checkpointWritesCollectionName,
      this.checkpointBlobsCollectionName,
    ]) {
      const { deletedCount } = await this.db
        .collection(name)
        .deleteMany({ thread_id });
      documents += deletedCount;
    }
    await this.db
      .collection(this.retention!.threadsCollectionName)
      .updateOne(
        { thread_id, tombstone: true },
        { $unset: { last_active: "" } }
      );
    return documents;
  }

  /**
   * Deletes every checkpoint, pending write and blob of the thread, in every
   * namespace, plus its retention record (REQ-E-070…072, INV-008).
   *
   * Order: manifests, then pending writes, then blobs, then the `threads`
   * record, so a partial failure never leaves a manifest without its blobs.
   * On a replica set or sharded cluster the deletes run in one transaction;
   * otherwise they run sequentially and a re-run finishes the job.
   */
  async deleteThread(threadId: string) {
    if (typeof threadId !== "string") {
      throw new Error("Invalid threadId: expected a string");
    }
    const names = [
      this.checkpointCollectionName,
      this.checkpointWritesCollectionName,
      this.checkpointBlobsCollectionName,
      ...(this.retention ? [this.retention.threadsCollectionName] : []),
    ];
    const filter = { thread_id: threadId };

    if ((await this.serverInfo()).supportsTransactions) {
      await this.client.withSession((session) =>
        session.withTransaction(async () => {
          for (const name of names) {
            await this.db.collection(name).deleteMany(filter, { session });
          }
        })
      );
      return;
    }
    for (const name of names) {
      await this.db.collection(name).deleteMany(filter);
    }
  }
}

/** Wire version of MongoDB 8.0, which added the client-level `bulkWrite`. */
export const MIN_CLIENT_BULK_WRITE_WIRE = 25;

export interface ServerInfo {
  maxWireVersion: number;
  /** Replica set or sharded cluster. */
  supportsTransactions: boolean;
}

interface WriteOp {
  filter: Document;
  update: Document;
}

type WriteKey = { task_path?: string; task_id: string; idx: number };

/** Marks a failure of the `threads` upsert in the parallel stage. */
class ThreadsWriteError extends Error {
  readonly isThreadsWriteError = true;

  constructor(readonly cause: unknown) {
    super((cause as Error)?.message ?? String(cause));
  }
}

function isThreadsWriteError(e: unknown): e is ThreadsWriteError {
  return (e as { isThreadsWriteError?: boolean })?.isThreadsWriteError === true;
}

function isThreadsDuplicateKey(e: unknown): boolean {
  if (!isThreadsWriteError(e)) return false;
  const c = e.cause as { code?: number; writeErrors?: unknown };
  if (c?.code === 11000) return true;
  // client bulkWrite: the first (threads) operation failed
  const writeErrors = (c as { writeErrors?: Map<number, { code?: number }> })
    ?.writeErrors;
  return writeErrors?.get?.(0)?.code === 11000;
}

/** `Binary.value()` (the driver) or a raw value (tests' mocks). */
function binaryValue(b: { value: (asRaw?: unknown) => unknown } | Uint8Array) {
  if (ArrayBuffer.isView(b)) return b as Uint8Array;
  return b.value("utf8" as unknown) as Uint8Array | string;
}
