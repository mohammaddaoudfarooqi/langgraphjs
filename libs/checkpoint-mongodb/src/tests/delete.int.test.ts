import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { MongoClient, type Db } from "mongodb";
import { emptyCheckpoint, uuid6 } from "@langchain/langgraph-checkpoint";
import type { Checkpoint } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../index.js";
import { dropTestDb, newClient, serverInfo, uniqueDbName } from "./int_utils.js";

const COLLECTIONS = ["checkpoints", "checkpoint_writes", "checkpoint_blobs", "threads"];

let client: MongoClient;
let isReplicaSet: boolean;
const dbName = uniqueDbName("delete");

beforeAll(async () => {
  client = newClient();
  await client.connect();
  ({ isReplicaSet } = await serverInfo(client));
  await new MongoDBSaver({ client, dbName, retention: { enabled: true } }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

async function populate(saver: MongoDBSaver, db: Db, thread_id: string) {
  for (const checkpoint_ns of ["", "sub:1", "sub:2|inner:3"]) {
    const v = saver.getNextVersion(undefined);
    const checkpoint: Checkpoint = {
      ...emptyCheckpoint(),
      id: uuid6(-1),
      channel_values: { a: { ns: checkpoint_ns }, b: [1, 2] },
      channel_versions: { a: v, b: v },
    };
    const config = await saver.put(
      { configurable: { thread_id, checkpoint_ns } },
      checkpoint,
      { source: "loop", step: 1, parents: {} },
      checkpoint.channel_versions
    );
    await saver.putWrites(config, [["a", 1], ["b", 2]], "task");
  }
  const counts: Record<string, number> = {};
  for (const c of COLLECTIONS) {
    counts[c] = await db.collection(c).countDocuments({ thread_id });
  }
  return counts;
}

async function counts(db: Db, thread_id: string) {
  const out: Record<string, number> = {};
  for (const c of COLLECTIONS) out[c] = await db.collection(c).countDocuments({ thread_id });
  return out;
}

/** Makes `deleteMany` on `failOn` throw (a failure after the first delete). */
function injectFailure(saver: MongoDBSaver, failOn: string) {
  const s = saver as unknown as { db: Db };
  const real = s.db;
  s.db = new Proxy(real, {
    get(target, prop, receiver) {
      if (prop === "collection") {
        return (name: string) => {
          const coll = target.collection(name);
          if (name !== failOn) return coll;
          return new Proxy(coll, {
            get(t, p, r) {
              if (p === "deleteMany") {
                return async () => {
                  throw new Error(`injected failure on ${name}`);
                };
              }
              return Reflect.get(t, p, r);
            },
          });
        };
      }
      return Reflect.get(target, prop, receiver);
    },
  });
  return () => {
    s.db = real;
  };
}

describe("deleteThread", () => {
  it("TC-070: deletes the thread from all four collections, across namespaces", async () => {
    const saver = new MongoDBSaver({ client, dbName, retention: { enabled: true } });
    const db = client.db(dbName);
    const thread_id = uuid6(-1);
    const other = uuid6(-1);
    const before = await populate(saver, db, thread_id);
    await populate(saver, db, other);
    expect(before).toEqual({
      checkpoints: 3,
      checkpoint_writes: 6,
      checkpoint_blobs: 6,
      threads: 1,
    });
    await saver.deleteThread(thread_id);
    expect(await counts(db, thread_id)).toEqual({
      checkpoints: 0,
      checkpoint_writes: 0,
      checkpoint_blobs: 0,
      threads: 0,
    });
    expect((await counts(db, other)).checkpoints).toBe(3);
    // the thread id can be reused after deleteThread
    await populate(saver, db, thread_id);
  });

  it("TC-071: on a replica set, a failure after the first delete rolls everything back", async (ctx) => {
    if (!isReplicaSet) ctx.skip();
    const saver = new MongoDBSaver({ client, dbName, retention: { enabled: true } });
    const db = client.db(dbName);
    const thread_id = uuid6(-1);
    const before = await populate(saver, db, thread_id);
    const restore = injectFailure(saver, "checkpoint_writes");
    await expect(saver.deleteThread(thread_id)).rejects.toThrow(/injected failure/);
    restore();
    expect(await counts(db, thread_id)).toEqual(before);
    expect(await saver.getTuple({ configurable: { thread_id } })).toBeDefined();
  });

  it("TC-072: without transactions, a re-run after a partial failure deletes the rest", async () => {
    // A real standalone server when MONGODB_STANDALONE_URL is set, otherwise a
    // saver that reports no transaction support on the current server.
    const standaloneUrl = process.env.MONGODB_STANDALONE_URL;
    const c = standaloneUrl ? new MongoClient(standaloneUrl) : client;
    const name = standaloneUrl ? uniqueDbName("delete_standalone") : dbName;
    class NoTransactions extends MongoDBSaver {
      protected async serverInfo() {
        return { ...(await super.serverInfo()), supportsTransactions: false };
      }
    }
    try {
      if (standaloneUrl) {
        expect((await serverInfo(c)).isReplicaSet).toBe(false);
      }
      const saver = standaloneUrl
        ? new MongoDBSaver({ client: c, dbName: name, retention: { enabled: true } })
        : new NoTransactions({ client: c, dbName: name, retention: { enabled: true } });
      await saver.setup();
      const db = c.db(name);
      const thread_id = uuid6(-1);
      await populate(saver, db, thread_id);
      const restore = injectFailure(saver, "checkpoint_blobs");
      await expect(saver.deleteThread(thread_id)).rejects.toThrow(/injected failure/);
      restore();
      // manifests and writes are gone (in order), blobs and the record remain
      const partial = await counts(db, thread_id);
      expect(partial.checkpoints).toBe(0);
      expect(partial.checkpoint_writes).toBe(0);
      expect(partial.checkpoint_blobs).toBe(6);
      await saver.deleteThread(thread_id);
      expect(await counts(db, thread_id)).toEqual({
        checkpoints: 0,
        checkpoint_writes: 0,
        checkpoint_blobs: 0,
        threads: 0,
      });
      await saver.deleteThread(thread_id); // idempotent
    } finally {
      if (standaloneUrl) {
        await dropTestDb(c, name);
        await c.close();
      }
    }
  });
});
