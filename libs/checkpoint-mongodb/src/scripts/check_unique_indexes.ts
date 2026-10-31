/**
 * Checks (and optionally rebuilds) the unique indexes required by
 * `MongoDBSaver` (REQ-E-022).
 *
 * Databases created by @langchain/langgraph-checkpoint-mongodb <= 1.4.x have
 * NON-unique compound indexes. `MongoDBSaver.setup()` reports those as
 * `IndexMigrationRequired` and never replaces them itself. This script:
 *
 *   1. reports duplicate logical keys in `checkpoints`, `checkpoint_writes` and
 *      `checkpoint_blobs` (which would make a unique index build fail);
 *   2. with `--apply`, and only when a collection has no duplicates, converts
 *      the existing index to unique in place (`collMod` `prepareUnique` then
 *      `unique`, MongoDB 6.0+), or creates it if it is missing.
 *
 * It is idempotent: re-running after a successful `--apply` reports `ok`.
 * Duplicates are never deleted automatically — resolve them by hand first.
 *
 * Usage (the connection string is read from MONGODB_URI or MONGODB_URL):
 *
 *   MONGODB_URI=... npx tsx scripts/check-unique-indexes.ts --db <name> \
 *     [--checkpoints checkpoints] [--writes checkpoint_writes] \
 *     [--blobs checkpoint_blobs] [--apply]
 *
 * Exit code: 0 when every index is unique after the run, 1 otherwise.
 */
import { MongoClient } from "mongodb";
import { requiredIndexes } from "../checkpoint.js";

export type IndexStatus =
  | "ok" // a unique index with the required keys exists
  | "missing" // no index with the required keys (report only)
  | "non-unique" // a non-unique index exists and there are no duplicates
  | "duplicates" // duplicate logical keys exist; nothing was changed
  | "created" // --apply created the missing unique index
  | "rebuilt"; // --apply converted the existing index to unique

export interface IndexReport {
  collection: string;
  indexName: string;
  keys: Record<string, number>;
  status: IndexStatus;
  duplicates: { key: Record<string, unknown>; count: number }[];
}

export interface CheckUniqueIndexesOptions {
  client: MongoClient;
  dbName?: string;
  checkpointCollectionName?: string;
  checkpointWritesCollectionName?: string;
  checkpointBlobsCollectionName?: string;
  apply?: boolean;
  /** Max duplicate groups listed per collection. Defaults to 100. */
  maxReported?: number;
}

function sameKeys(a: Record<string, unknown>, b: Record<string, unknown>) {
  const ak = Object.entries(a);
  const bk = Object.entries(b);
  return (
    ak.length === bk.length &&
    ak.every(([k, v], i) => bk[i][0] === k && Number(bk[i][1]) === Number(v))
  );
}

export async function checkUniqueIndexes(
  options: CheckUniqueIndexesOptions
): Promise<IndexReport[]> {
  const db = options.client.db(options.dbName);
  const specs = requiredIndexes({
    checkpoints: options.checkpointCollectionName ?? "checkpoints",
    writes: options.checkpointWritesCollectionName ?? "checkpoint_writes",
    blobs: options.checkpointBlobsCollectionName ?? "checkpoint_blobs",
  }).filter((s) => s.unique);

  const reports: IndexReport[] = [];
  for (const spec of specs) {
    const coll = db.collection(spec.collection);
    const indexName = spec.options.name as string;
    const existing = (
      await coll
        .listIndexes()
        .toArray()
        .catch((e) => {
          if ((e as { code?: number }).code === 26) return [];
          throw e;
        })
    ).find((idx) => sameKeys(idx.key, spec.keys));

    const report: IndexReport = {
      collection: spec.collection,
      indexName: existing?.name ?? indexName,
      keys: spec.keys,
      status: "ok",
      duplicates: [],
    };
    reports.push(report);
    if (existing?.unique === true) continue;

    const group: Record<string, string> = {};
    for (const k of Object.keys(spec.keys)) group[k] = `$${k}`;
    report.duplicates = (
      await coll
        .aggregate(
          [
            { $group: { _id: group, count: { $sum: 1 } } },
            { $match: { count: { $gt: 1 } } },
            { $sort: { count: -1 } },
            { $limit: options.maxReported ?? 100 },
          ],
          { allowDiskUse: true }
        )
        .toArray()
    ).map((d) => ({ key: d._id as Record<string, unknown>, count: d.count }));

    if (report.duplicates.length > 0) {
      report.status = "duplicates";
      continue;
    }
    if (existing === undefined) {
      report.status = "missing";
      if (options.apply) {
        await coll.createIndex(spec.keys, spec.options);
        report.status = "created";
      }
      continue;
    }
    report.status = "non-unique";
    if (!options.apply) continue;

    // Convert in place: prepareUnique rejects new duplicates, then unique:true
    // validates existing documents and flips the index (MongoDB 6.0+).
    const index = { name: existing.name };
    await db.command({
      collMod: spec.collection,
      index: { ...index, prepareUnique: true },
    });
    try {
      await db.command({
        collMod: spec.collection,
        index: { ...index, unique: true },
      });
    } catch (e) {
      await db
        .command({
          collMod: spec.collection,
          index: { ...index, prepareUnique: false },
        })
        .catch(() => {});
      throw e;
    }
    report.status = "rebuilt";
  }
  return reports;
}

/** CLI entry point used by `scripts/check-unique-indexes.ts`. */
export async function checkUniqueIndexesCli(args: string[]): Promise<number> {
  const flag = (name: string) => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const uri = process.env.MONGODB_URI ?? process.env.MONGODB_URL;
  if (!uri) {
    console.error("Set MONGODB_URI (or MONGODB_URL) to the connection string.");
    return 2;
  }
  const client = new MongoClient(uri);
  try {
    const reports = await checkUniqueIndexes({
      client,
      dbName: flag("db"),
      checkpointCollectionName: flag("checkpoints"),
      checkpointWritesCollectionName: flag("writes"),
      checkpointBlobsCollectionName: flag("blobs"),
      apply: args.includes("--apply"),
    });
    console.log(JSON.stringify(reports, null, 2));
    return reports.every((r) => ["ok", "created", "rebuilt"].includes(r.status))
      ? 0
      : 1;
  } finally {
    await client.close();
  }
}
