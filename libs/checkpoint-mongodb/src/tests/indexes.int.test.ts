import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import { MongoDBSaver, IndexMigrationRequired } from "../index.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

let client: MongoClient;
const dbNames: string[] = [];
function freshDb(label: string) {
  const name = uniqueDbName(label);
  dbNames.push(name);
  return name;
}

beforeAll(async () => {
  client = newClient();
  await client.connect();
});

afterAll(async () => {
  for (const name of dbNames) await dropTestDb(client, name);
  await client.close();
});

async function indexes(dbName: string, coll: string) {
  return client
    .db(dbName)
    .collection(coll)
    .listIndexes()
    .toArray()
    .catch(() => []);
}

describe("unique indexes", () => {
  it("TC-020: setup creates the three unique compound indexes, no threads indexes without retention", async () => {
    const dbName = freshDb("idx_a");
    const saver = new MongoDBSaver({ client, dbName });
    expect(await saver.setup()).toEqual([]);

    const cp = await indexes(dbName, "checkpoints");
    expect(cp).toContainEqual(
      expect.objectContaining({
        key: { thread_id: 1, checkpoint_ns: 1, checkpoint_id: -1 },
        unique: true,
      })
    );
    const w = await indexes(dbName, "checkpoint_writes");
    expect(w).toContainEqual(
      expect.objectContaining({
        key: {
          thread_id: 1,
          checkpoint_ns: 1,
          checkpoint_id: 1,
          task_id: 1,
          idx: 1,
        },
        unique: true,
      })
    );
    const b = await indexes(dbName, "checkpoint_blobs");
    expect(b).toContainEqual(
      expect.objectContaining({
        key: { thread_id: 1, checkpoint_ns: 1, channel: 1, version: 1 },
        unique: true,
      })
    );
    expect(await indexes(dbName, "threads")).toEqual([]);
  });

  it("TC-020: with retention, setup also creates the threads indexes", async () => {
    const dbName = freshDb("idx_b");
    const saver = new MongoDBSaver({
      client,
      dbName,
      retention: { enabled: true, tombstoneSeconds: 3600 },
    });
    expect(await saver.setup()).toEqual([]);
    const t = await indexes(dbName, "threads");
    expect(t).toContainEqual(
      expect.objectContaining({ key: { thread_id: 1 }, unique: true })
    );
    expect(t).toContainEqual(
      expect.objectContaining({
        key: { tombstoned_at: 1 },
        expireAfterSeconds: 3600,
      })
    );
  });

  it("TC-021: an existing non-unique index is reported, not replaced", async () => {
    const dbName = freshDb("idx_c");
    const coll = client.db(dbName).collection("checkpoints");
    await coll.createIndex(
      { thread_id: 1, checkpoint_ns: 1, checkpoint_id: -1 },
      { name: "thread_ns_checkpoint_idx" }
    );
    const before = await indexes(dbName, "checkpoints");

    const saver = new MongoDBSaver({ client, dbName });
    const errors = await saver.setup();
    expect(errors).toHaveLength(1);
    const [err] = errors as IndexMigrationRequired[];
    expect(err).toBeInstanceOf(IndexMigrationRequired);
    expect(err.code).toBe("IndexMigrationRequired");
    expect(err.collection).toBe("checkpoints");
    expect(err.indexName).toBe("thread_ns_checkpoint_idx");
    expect(err.keys).toEqual({ thread_id: 1, checkpoint_ns: 1, checkpoint_id: -1 });

    expect(await indexes(dbName, "checkpoints")).toEqual(before);
    // The other collections are still set up.
    expect(await indexes(dbName, "checkpoint_writes")).toContainEqual(
      expect.objectContaining({ unique: true })
    );
  });

  it("INV-010: concurrent setup returns no errors", async () => {
    const dbName = freshDb("idx_d");
    const savers = Array.from(
      { length: 4 },
      () =>
        new MongoDBSaver({
          client,
          dbName,
          retention: { enabled: true },
        })
    );
    const results = await Promise.all(savers.map((s) => s.setup()));
    expect(results).toEqual([[], [], [], []]);
    expect(await savers[0].setup()).toEqual([]);
  });
});
