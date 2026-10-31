import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import type { MongoClient } from "mongodb";
import { emptyCheckpoint, uuid6 } from "@langchain/langgraph-checkpoint";
import type { Checkpoint } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver, ThreadExpiredError } from "../index.js";
import { _resetDeprecationWarnings } from "../checkpoint.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

let client: MongoClient;
const dbName = uniqueDbName("retention");

beforeAll(async () => {
  client = newClient();
  await client.connect();
  await new MongoDBSaver({ client, dbName, retention: { enabled: true } }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

const coll = (name: string) => client.db(dbName).collection(name);
const retained = () =>
  new MongoDBSaver({ client, dbName, retention: { enabled: true } });

async function put(saver: MongoDBSaver, thread_id: string, parent?: string) {
  const v = saver.getNextVersion(undefined);
  const checkpoint: Checkpoint = {
    ...emptyCheckpoint(),
    id: uuid6(-1),
    channel_values: { a: { v: Math.random() }, s: "inline" },
    channel_versions: { a: v, s: v },
  };
  const config = await saver.put(
    { configurable: { thread_id, checkpoint_ns: "", checkpoint_id: parent } },
    checkpoint,
    { source: "loop", step: 1, parents: {} },
    checkpoint.channel_versions
  );
  return { checkpoint, config };
}

/** Forces the pre-8.0 write path (no client-level bulkWrite). */
class Pre80Saver extends MongoDBSaver {
  protected async serverInfo() {
    return { ...(await super.serverInfo()), maxWireVersion: 21 };
  }
}

async function age(thread_id: string) {
  await coll("threads").updateOne(
    { thread_id },
    { $set: { last_active: new Date(0) } }
  );
}

describe("thread-scoped retention", () => {
  it("TC-060: put/putWrites create and advance threads.last_active; off by default", async () => {
    const thread_id = uuid6(-1);
    const { config } = await put(retained(), thread_id);
    const first = await coll("threads").findOne({ thread_id });
    expect(first).toMatchObject({ thread_id, generation: 0 });
    expect(first!.last_active).toBeInstanceOf(Date);
    expect(first!.tombstone).toBeUndefined();
    await new Promise((r) => setTimeout(r, 5));
    await retained().putWrites(config, [["a", 1]], "task");
    const second = await coll("threads").findOne({ thread_id });
    expect(second!.last_active.getTime()).toBeGreaterThan(first!.last_active.getTime());
    // an empty putWrites touches nothing (INV-004)
    await retained().putWrites(config, [], "task");

    const offDb = uniqueDbName("retention_off");
    try {
      const off = new MongoDBSaver({ client, dbName: offDb });
      const { config: c2 } = await put(off, uuid6(-1));
      await off.putWrites(c2, [["a", 1]], "task");
      expect(await client.db(offDb).collection("threads").countDocuments()).toBe(0);
    } finally {
      await dropTestDb(client, offDb);
    }
  });

  it("TC-061: an inactive thread is swept; the tombstone remains and blocks writes", async () => {
    const thread_id = uuid6(-1);
    const keep = uuid6(-1);
    const { config } = await put(retained(), thread_id);
    await put(retained(), thread_id, config.configurable!.checkpoint_id);
    await retained().putWrites(config, [["a", 1]], "task");
    await put(retained(), keep);
    await age(thread_id);

    const result = await retained().sweepExpired({
      olderThan: new Date(Date.now() - 60_000),
    });
    expect(result.threads).toBe(1);
    expect(result.documents).toBe(2 + 1 + 2); // manifests + writes + blobs
    for (const c of ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]) {
      expect(await coll(c).countDocuments({ thread_id })).toBe(0);
    }
    for (const c of ["checkpoints", "checkpoint_blobs"]) {
      expect(await coll(c).countDocuments({ thread_id: keep })).toBeGreaterThan(0);
    }
    const tomb = await coll("threads").findOne({ thread_id });
    expect(tomb).toMatchObject({ thread_id, tombstone: true, generation: 1 });
    expect(tomb!.tombstoned_at).toBeInstanceOf(Date);

    await expect(put(retained(), thread_id)).rejects.toBeInstanceOf(ThreadExpiredError);
    const pre80 = new Pre80Saver({ client, dbName, retention: { enabled: true } });
    await expect(put(pre80, thread_id)).rejects.toBeInstanceOf(ThreadExpiredError);
    // nothing may outlive the purge: no sweep would ever remove it
    expect(await coll("checkpoint_blobs").countDocuments({ thread_id })).toBe(0);
    await expect(
      retained().putWrites(config, [["a", 2]], "task2")
    ).rejects.toBeInstanceOf(ThreadExpiredError);
    expect(await coll("checkpoints").countDocuments({ thread_id })).toBe(0);
    expect(await coll("checkpoint_writes").countDocuments({ thread_id })).toBe(0);
    // re-running finds nothing more to do
    expect(
      await retained().sweepExpired({ olderThan: new Date(Date.now() - 60_000) })
    ).toEqual({ threads: 0, documents: 0 });
  });

  it("TC-062: a sweep racing a put never leaves a manifest without its blobs (50 iterations)", async () => {
    let committed = 0;
    let expired = 0;
    for (let i = 0; i < 50; i += 1) {
      const thread_id = uuid6(-1);
      const { config } = await put(retained(), thread_id);
      await age(thread_id);
      const olderThan = new Date(Date.now() - 1_000);
      const newId = { value: "" };
      const [sweep, write] = await Promise.allSettled([
        retained().sweepExpired({ olderThan }),
        // vary which side wins the race
        (i % 2 === 0 ? Promise.resolve() : new Promise((r) => setTimeout(r, i % 3)))
          .then(() => put(retained(), thread_id, config.configurable!.checkpoint_id))
          .then((r) => {
          newId.value = r.checkpoint.id;
          return r;
        }),
      ]);
      expect(sweep.status).toBe("fulfilled");
      const record = await coll("threads").findOne({ thread_id });
      if (write.status === "fulfilled") {
        committed += 1;
        // the write committed: the thread must not have been swept
        expect(record!.tombstone).toBeUndefined();
        const tuple = await retained().getTuple({ configurable: { thread_id } });
        expect(tuple!.checkpoint.id).toBe(newId.value);
      } else {
        expired += 1;
        expect(write.reason).toBeInstanceOf(ThreadExpiredError);
        expect(record!.tombstone).toBe(true);
        expect(await coll("checkpoints").countDocuments({ thread_id })).toBe(0);
      }
      // never a manifest whose blobs are gone
      for await (const _ of retained().list({ configurable: { thread_id } })) {
        // list() raises CheckpointIntegrityError on a missing blob
      }
    }
    expect(committed + expired).toBe(50);
    process.stderr.write(`TC-062: ${committed} writes committed, ${expired} expired\n`);
  }, 600_000);

  it("TC-063: a sweep interrupted after the claim completes on re-run and returns counts", async () => {
    const thread_id = uuid6(-1);
    const { config } = await put(retained(), thread_id);
    await retained().putWrites(config, [["a", 1], ["b", 2]], "task");
    await age(thread_id);
    // What sweepExpired's claim does, without the deletes that follow it.
    await coll("threads").updateOne(
      { thread_id, tombstone: { $exists: false } },
      { $set: { tombstone: true, tombstoned_at: new Date() }, $inc: { generation: 1 } }
    );
    expect(await coll("checkpoints").countDocuments({ thread_id })).toBe(1);

    const result = await retained().sweepExpired({
      olderThan: new Date(Date.now() - 60_000),
    });
    expect(result).toEqual({ threads: 1, documents: 1 + 2 + 1 });
    for (const c of ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]) {
      expect(await coll(c).countDocuments({ thread_id })).toBe(0);
    }
    expect(
      await retained().sweepExpired({ olderThan: new Date(Date.now() - 60_000) })
    ).toEqual({ threads: 0, documents: 0 });
  });

  it("TC-064: the old ttl option logs one deprecation warning and keeps working", async () => {
    _resetDeprecationWarnings();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const a = new MongoDBSaver({ client, dbName, ttl: 3600 });
      // eslint-disable-next-line no-new
      new MongoDBSaver({ client, dbName, ttl: 60 });
      const ttlWarnings = warn.mock.calls.filter((c) => String(c[0]).includes("ttl"));
      expect(ttlWarnings).toHaveLength(1);
      expect(String(ttlWarnings[0][0])).toContain("sweepExpired");
      // INV-009: upserted_at is still written
      const thread_id = uuid6(-1);
      await a.put(
        { configurable: { thread_id } },
        { ...emptyCheckpoint(), id: uuid6(-1), channel_values: { x: 1 }, channel_versions: { x: 1 } },
        { source: "loop", step: 1, parents: {} },
        { x: 1 }
      );
      expect((await coll("checkpoints").findOne({ thread_id }))!.upserted_at).toBeInstanceOf(Date);
    } finally {
      warn.mockRestore();
    }
  });
});
