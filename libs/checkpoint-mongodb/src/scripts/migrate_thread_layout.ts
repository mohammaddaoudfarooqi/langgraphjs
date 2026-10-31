/**
 * Converts selected old-format threads to the per-channel layout (layout 2)
 * (REQ-E-042). Optional: old-format threads keep working without it.
 *
 * For each thread (all namespaces):
 *
 *   1. every old-format checkpoint is converted — integer channel versions
 *      become `"{n:032}.{suffix}"` strings with the same integer prefix, so
 *      ordering is preserved — and written as a layout 2 manifest to a
 *      temporary staging collection, with its blobs in `checkpoint_blobs`;
 *   2. every converted checkpoint is read back and compared with the
 *      original (values, metadata, parents, version prefixes);
 *   3. only then are the originals copied to a backup collection and replaced
 *      in place by the converted manifests. On a verification failure the
 *      originals are left untouched. The backup is removed once the live
 *      checkpoints read back correctly, unless `keepBackup` is set.
 *
 * The version suffix is derived from a hash of the channel's serialized value,
 * so the same `(channel, n)` with the same value maps to the same version
 * (blobs are shared across checkpoints), while fork branches that reached the
 * same integer version with different values get distinct versions.
 *
 * Re-running on a migrated thread is a no-op. Do not run it while the thread
 * is being written to.
 *
 * Usage (the connection string is read from MONGODB_URI or MONGODB_URL):
 *
 *   MONGODB_URI=... npx tsx scripts/migrate-thread-layout.ts --db <name> \
 *     --thread <thread_id> [--thread <thread_id> ...]
 */
import { createHash, randomBytes } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { MongoClient, type Document } from "mongodb";
import type {
  Checkpoint,
  CheckpointTuple,
  SerializerProtocol,
} from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../checkpoint.js";
import { isLayout2 } from "../layout.js";
import { formatVersion, versionPrefix } from "../versions.js";

export type MigrationStatus =
  | "migrated"
  | "already-migrated"
  | "not-found"
  | "verification-failed";

export interface MigrationResult {
  threadId: string;
  status: MigrationStatus;
  /** Number of checkpoints converted. */
  checkpoints: number;
  /** Collection holding copies of the replaced old-format documents (only with `keepBackup`). */
  backupCollection?: string;
  error?: string;
}

export interface MigrateThreadLayoutOptions {
  client: MongoClient;
  dbName?: string;
  threadIds: string[];
  checkpointCollectionName?: string;
  checkpointWritesCollectionName?: string;
  checkpointBlobsCollectionName?: string;
  /** Serializer the threads were written with (default: the saver's). */
  serde?: SerializerProtocol;
  /** Defaults to `<checkpoints>_layout1_backup`. */
  backupCollectionName?: string;
  /**
   * Keep the backup copies after a successful migration. They are outside
   * `deleteThread` and retention, so by default they are removed.
   */
  keepBackup?: boolean;
  /** Test hook, called after conversion and before verification. */
  beforeVerify?: (collections: {
    staging: string;
    blobs: string;
  }) => Promise<void>;
}

/** Writes and reads layout 2 for every checkpoint, including integer versions. */
class Layout2Saver extends MongoDBSaver {
  protected useLayout2(): boolean {
    return true;
  }
}

const ZERO_SUFFIX = "0.00000000000000";

/** Deterministic 16-character suffix ("0." + 14 digits) for a channel value. */
function hashSuffix(parts: (string | Uint8Array)[]): string {
  const h = createHash("sha256");
  for (const p of parts) {
    h.update(p);
    h.update("\0");
  }
  const n = BigInt(`0x${h.digest("hex").slice(0, 14)}`) % 10n ** 14n;
  return `0.${n.toString().padStart(14, "0")}`;
}

async function convertCheckpoint(
  checkpoint: Checkpoint,
  checkpointNs: string,
  serde: SerializerProtocol
): Promise<Checkpoint> {
  const mapped: Record<string, string | number> = {};
  for (const [ch, v] of Object.entries(checkpoint.channel_versions)) {
    if (typeof v !== "number") {
      mapped[ch] = v;
      continue;
    }
    let parts: (string | Uint8Array)[] = [checkpointNs, ch, String(v)];
    if (Object.prototype.hasOwnProperty.call(checkpoint.channel_values, ch)) {
      const [type, bytes] = await serde.dumpsTyped(
        checkpoint.channel_values[ch]
      );
      parts = [...parts, type, bytes];
    }
    mapped[ch] = formatVersion(v, hashSuffix(parts));
  }
  const versionsSeen: Checkpoint["versions_seen"] = {};
  for (const [node, seen] of Object.entries(checkpoint.versions_seen ?? {})) {
    versionsSeen[node] = {};
    for (const [ch, m] of Object.entries(seen)) {
      if (typeof m !== "number") {
        versionsSeen[node][ch] = m;
      } else if (m === checkpoint.channel_versions[ch]) {
        // "seen the current version" must stay an exact match
        versionsSeen[node][ch] = mapped[ch];
      } else {
        versionsSeen[node][ch] = formatVersion(m, ZERO_SUFFIX);
      }
    }
  }
  return {
    ...checkpoint,
    channel_versions: mapped,
    versions_seen: versionsSeen,
  };
}

/** Replaces every version by its integer prefix (for verification). */
function versionPrefixes(checkpoint: Checkpoint) {
  const prefix = (v: unknown) =>
    typeof v === "number" ? v : versionPrefix(String(v));
  return {
    ...checkpoint,
    channel_versions: Object.fromEntries(
      Object.entries(checkpoint.channel_versions).map(([k, v]) => [
        k,
        prefix(v),
      ])
    ),
    versions_seen: Object.fromEntries(
      Object.entries(checkpoint.versions_seen ?? {}).map(([node, seen]) => [
        node,
        Object.fromEntries(
          Object.entries(seen).map(([k, v]) => [k, prefix(v)])
        ),
      ])
    ),
  };
}

function sameTuple(a: CheckpointTuple, b: CheckpointTuple): string | undefined {
  if (
    !isDeepStrictEqual(a.checkpoint.channel_values, b.checkpoint.channel_values)
  )
    return "channel_values differ";
  if (
    !isDeepStrictEqual(
      versionPrefixes(a.checkpoint),
      versionPrefixes(b.checkpoint)
    )
  )
    return "checkpoint fields or version prefixes differ";
  if (!isDeepStrictEqual(a.metadata, b.metadata)) return "metadata differs";
  if (!isDeepStrictEqual(a.parentConfig, b.parentConfig))
    return "parent differs";
  if (!isDeepStrictEqual(a.pendingWrites, b.pendingWrites))
    return "pending writes differ";
  return undefined;
}

export async function migrateThreadLayout(
  options: MigrateThreadLayoutOptions
): Promise<MigrationResult[]> {
  const results: MigrationResult[] = [];
  for (const threadId of options.threadIds) {
    results.push(await migrateOne(options, threadId));
  }
  return results;
}

async function migrateOne(
  options: MigrateThreadLayoutOptions,
  threadId: string
): Promise<MigrationResult> {
  const { client } = options;
  const db = client.db(options.dbName);
  const names = {
    checkpointCollectionName: options.checkpointCollectionName ?? "checkpoints",
    checkpointWritesCollectionName:
      options.checkpointWritesCollectionName ?? "checkpoint_writes",
    checkpointBlobsCollectionName:
      options.checkpointBlobsCollectionName ?? "checkpoint_blobs",
  };
  const checkpoints = db.collection(names.checkpointCollectionName);

  const docs = await checkpoints
    .find({ thread_id: threadId })
    .sort({ checkpoint_ns: 1, checkpoint_id: 1 })
    .toArray();
  if (docs.length === 0) {
    return { threadId, status: "not-found", checkpoints: 0 };
  }
  const oldDocs = docs.filter((d) => !isLayout2(d));
  if (oldDocs.length === 0) {
    return { threadId, status: "already-migrated", checkpoints: 0 };
  }

  const reader = new MongoDBSaver(
    { client, dbName: options.dbName, ...names },
    options.serde
  );
  const serde = (reader as unknown as { baseSerde: SerializerProtocol })
    .baseSerde;
  const staging = `${names.checkpointCollectionName}__migrate_${randomBytes(
    4
  ).toString("hex")}`;
  const stagingSaver = new Layout2Saver(
    {
      client,
      dbName: options.dbName,
      ...names,
      checkpointCollectionName: staging,
    },
    options.serde
  );
  const cfg = (d: Document) => ({
    configurable: {
      thread_id: threadId,
      checkpoint_ns: d.checkpoint_ns as string,
      checkpoint_id: d.checkpoint_id as string,
    },
  });

  const written = new Set<string>();
  const originals = new Map<string, CheckpointTuple>();
  try {
    // 1. convert into the staging collection
    for (const doc of oldDocs) {
      const tuple = (await reader.getTuple(cfg(doc)))!;
      originals.set(String(doc._id), tuple);
      const converted = await convertCheckpoint(
        tuple.checkpoint,
        doc.checkpoint_ns,
        serde
      );
      const newVersions: Record<string, string | number> = {};
      for (const [ch, v] of Object.entries(converted.channel_versions)) {
        const key = JSON.stringify([doc.checkpoint_ns, ch, v]);
        if (!written.has(key)) {
          written.add(key);
          newVersions[ch] = v;
        }
      }
      await stagingSaver.put(
        {
          configurable: {
            thread_id: threadId,
            checkpoint_ns: doc.checkpoint_ns,
            checkpoint_id: doc.parent_checkpoint_id ?? undefined,
          },
        },
        converted,
        tuple.metadata!,
        newVersions
      );
    }

    await options.beforeVerify?.({
      staging,
      blobs: names.checkpointBlobsCollectionName,
    });

    // 2. verify every converted checkpoint against its original
    for (const doc of oldDocs) {
      const original = originals.get(String(doc._id))!;
      let problem: string | undefined;
      try {
        const converted = await stagingSaver.getTuple(cfg(doc));
        problem =
          converted === undefined
            ? "converted checkpoint missing"
            : sameTuple(original, converted);
      } catch (e) {
        problem = (e as Error).message;
      }
      if (problem !== undefined) {
        throw new VerificationError(
          `checkpoint ${doc.checkpoint_id} (ns "${doc.checkpoint_ns}"): ${problem}`
        );
      }
    }

    // 3. back up the originals, then replace them in place
    const backupName =
      options.backupCollectionName ??
      `${names.checkpointCollectionName}_layout1_backup`;
    const backup = db.collection(backupName);
    for (const doc of oldDocs) {
      await backup.replaceOne({ _id: doc._id }, doc, { upsert: true });
    }
    for (const doc of oldDocs) {
      const converted = await db.collection(staging).findOne({
        thread_id: threadId,
        checkpoint_ns: doc.checkpoint_ns,
        checkpoint_id: doc.checkpoint_id,
      });
      const { _id, ...replacement } = converted!;
      if (doc.upserted_at !== undefined)
        replacement.upserted_at = doc.upserted_at;
      await checkpoints.replaceOne(
        { _id: doc._id, layout: { $exists: false } },
        replacement
      );
    }

    // 4. the live collection now returns the converted checkpoints
    for (const doc of oldDocs) {
      const live = await reader.getTuple(cfg(doc));
      const problem =
        live === undefined
          ? "missing after swap"
          : sameTuple(originals.get(String(doc._id))!, live);
      if (problem !== undefined) {
        throw new Error(
          `post-swap check failed for ${doc.checkpoint_id}: ${problem}; ` +
            `originals are in ${backupName}`
        );
      }
    }
    if (!options.keepBackup) {
      await backup.deleteMany({ _id: { $in: oldDocs.map((d) => d._id) } });
    }
    return {
      threadId,
      status: "migrated",
      checkpoints: oldDocs.length,
      ...(options.keepBackup ? { backupCollection: backupName } : {}),
    };
  } catch (e) {
    if (!(e instanceof VerificationError)) throw e;
    // New blobs are left in place: their keys are deterministic, so a re-run
    // may share them with checkpoints that were already swapped.
    return {
      threadId,
      status: "verification-failed",
      checkpoints: 0,
      error: e.message,
    };
  } finally {
    await db.dropCollection(staging).catch(() => {});
  }
}

class VerificationError extends Error {}

/** CLI entry point used by `scripts/migrate-thread-layout.ts`. */
export async function migrateThreadLayoutCli(args: string[]): Promise<number> {
  const values = (name: string) =>
    args.flatMap((a, i) => (a === `--${name}` ? [args[i + 1]] : []));
  const uri = process.env.MONGODB_URI ?? process.env.MONGODB_URL;
  if (!uri) {
    console.error("Set MONGODB_URI (or MONGODB_URL) to the connection string.");
    return 2;
  }
  const threadIds = values("thread");
  if (threadIds.length === 0) {
    console.error("Pass at least one --thread <thread_id>.");
    return 2;
  }
  const client = new MongoClient(uri);
  try {
    const results = await migrateThreadLayout({
      client,
      dbName: values("db")[0],
      threadIds,
      checkpointCollectionName: values("checkpoints")[0],
      checkpointWritesCollectionName: values("writes")[0],
      checkpointBlobsCollectionName: values("blobs")[0],
      keepBackup: args.includes("--keep-backup"),
    });
    console.log(JSON.stringify(results, null, 2));
    return results.every((r) =>
      ["migrated", "already-migrated"].includes(r.status)
    )
      ? 0
      : 1;
  } finally {
    await client.close();
  }
}
