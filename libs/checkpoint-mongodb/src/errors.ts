/**
 * Error types raised by the MongoDB checkpointer.
 *
 * Every class carries a `code` string equal to its class name so callers can
 * branch on errors without `instanceof` (e.g. across package copies). The
 * Python saver raises classes with the same names, `code` values and payload
 * fields (snake_case there).
 */
export class MongoDBCheckpointError extends Error {
  readonly code: string = "MongoDBCheckpointError";

  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = new.target.name;
    if (options?.cause !== undefined) {
      (this as { cause?: unknown }).cause = options.cause;
    }
  }
}

export interface ChannelSize {
  channel: string;
  bytes: number;
}

export interface DocumentSizeInfo {
  /** Target collection of the document. */
  collection: string;
  /** Exact BSON size of the document, in bytes. */
  bytes: number;
  /** Configured `maxDocumentBytes`. */
  limit: number;
  /**
   * Up to five largest channels (serialized bytes), largest first. Only
   * present when the document is above the warning threshold or the limit.
   */
  largestChannels?: ChannelSize[];
}

/** A checkpoint, blob or pending-write document is larger than `maxDocumentBytes`. */
export class CheckpointSizeError extends MongoDBCheckpointError {
  readonly code = "CheckpointSizeError";

  readonly collection: string;

  readonly bytes: number;

  readonly limit: number;

  readonly largestChannels: ChannelSize[];

  constructor(info: Required<DocumentSizeInfo>) {
    const top = info.largestChannels
      .map((c) => `${c.channel}=${c.bytes}`)
      .join(", ");
    super(
      `Document for collection "${info.collection}" is ${info.bytes} bytes, ` +
        `which exceeds the limit of ${info.limit} bytes. Nothing was written. ` +
        `Largest channels: ${top || "(none)"}`
    );
    this.collection = info.collection;
    this.bytes = info.bytes;
    this.limit = info.limit;
    this.largestChannels = info.largestChannels;
  }
}

/** A layout 2 checkpoint references blobs (or ancestors) that do not exist. */
export class CheckpointIntegrityError extends MongoDBCheckpointError {
  readonly code = "CheckpointIntegrityError";

  readonly threadId: string;

  readonly checkpointNs: string;

  readonly checkpointId: string;

  readonly missing: unknown[];

  constructor(params: {
    threadId: string;
    checkpointNs: string;
    checkpointId: string;
    missing: unknown[];
    message?: string;
  }) {
    super(
      params.message ??
        `Checkpoint ${params.checkpointId} (thread "${params.threadId}", ` +
          `namespace "${params.checkpointNs}") is incomplete; missing: ` +
          `${JSON.stringify(params.missing)}`
    );
    this.threadId = params.threadId;
    this.checkpointNs = params.checkpointNs;
    this.checkpointId = params.checkpointId;
    this.missing = params.missing;
  }
}

/** A write targeted a thread that retention has tombstoned. */
export class ThreadExpiredError extends MongoDBCheckpointError {
  readonly code = "ThreadExpiredError";

  readonly threadId: string;

  constructor(threadId: string, options?: { cause?: unknown }) {
    super(
      `Thread "${threadId}" has expired and was deleted by retention; ` +
        `it can no longer be written to.`,
      options
    );
    this.threadId = threadId;
  }
}

/** `setup()` found a non-unique index where a unique one is required. */
export class IndexMigrationRequired extends MongoDBCheckpointError {
  readonly code = "IndexMigrationRequired";

  readonly collection: string;

  readonly indexName: string;

  readonly keys: Record<string, number>;

  constructor(params: {
    collection: string;
    indexName: string;
    keys: Record<string, number>;
  }) {
    super(
      `Index "${params.indexName}" on collection "${params.collection}" ` +
        `(${JSON.stringify(params.keys)}) is not unique. Run ` +
        `scripts/check-unique-indexes.ts to check for duplicates and rebuild ` +
        `it as unique (--apply).`
    );
    this.collection = params.collection;
    this.indexName = params.indexName;
    this.keys = params.keys;
  }
}

/** The default serializer could not serialize a value (JS only). */
export class CheckpointSerializationError extends MongoDBCheckpointError {
  readonly code = "CheckpointSerializationError";
}
