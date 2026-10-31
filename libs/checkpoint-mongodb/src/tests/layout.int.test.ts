import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import {
  emptyCheckpoint,
  uuid6,
  type Checkpoint,
  type ChannelVersions,
} from "@langchain/langgraph-checkpoint";
import type { RunnableConfig } from "@langchain/core/runnables";
import {
  MongoDBSaver,
  CheckpointIntegrityError,
  CheckpointSizeError,
} from "../index.js";
import type { DocumentSizeInfo } from "../index.js";
import { VERSION_PATTERN } from "../versions.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

const MiB = 1024 * 1024;
const meta = { source: "loop" as const, step: 1, parents: {} };

let client: MongoClient;
const dbName = uniqueDbName("layout");

beforeAll(async () => {
  client = newClient();
  await client.connect();
  await new MongoDBSaver({ client, dbName }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

/**
 * Simulates the pregel loop: each step bumps the versions of the channels in
 * `changes` with `saver.getNextVersion` and passes them as `newVersions`.
 */
class ThreadDriver {
  config: RunnableConfig;

  values: Record<string, unknown> = {};

  versions: ChannelVersions = {};

  constructor(
    public saver: MongoDBSaver,
    threadId: string,
    checkpointNs = ""
  ) {
    this.config = { configurable: { thread_id: threadId, checkpoint_ns: checkpointNs } };
  }

  async step(changes: Record<string, unknown>) {
    const max = Object.values(this.versions).sort().pop();
    const newVersions: ChannelVersions = {};
    for (const [k, v] of Object.entries(changes)) {
      this.values[k] = v;
      newVersions[k] = this.saver.getNextVersion(max as never);
      this.versions[k] = newVersions[k];
    }
    const checkpoint: Checkpoint = {
      ...emptyCheckpoint(),
      id: uuid6(-1),
      channel_values: { ...this.values },
      channel_versions: { ...this.versions },
    };
    this.config = await this.saver.put(this.config, checkpoint, meta, newVersions);
    return checkpoint;
  }
}

const coll = (name: string) => client.db(dbName).collection(name);

describe("layout 2 write/read path", () => {
  it("TC-030: put writes a layout 2 manifest and the expected blobs", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    const cp = await t.step({
      small: "s",
      num: 3,
      flag: true,
      nil: null,
      obj: { a: 1 },
      list: [1, 2, 3],
      long: "x".repeat(2000),
    });

    const manifest = await coll("checkpoints").findOne({ thread_id });
    expect(manifest).not.toBeNull();
    expect(manifest!.layout).toBe(2);
    expect(manifest!.checkpoint_id).toBe(cp.id);
    expect(manifest!.inline_channels.sort()).toEqual(
      ["flag", "nil", "num", "small"].sort()
    );
    expect(manifest!.blob_refs).toEqual([
      { channel: "obj", version: cp.channel_versions.obj },
      { channel: "list", version: cp.channel_versions.list },
      { channel: "long", version: cp.channel_versions.long },
    ]);
    for (const v of Object.values(cp.channel_versions)) {
      expect(v).toMatch(VERSION_PATTERN);
    }

    const blobs = await coll("checkpoint_blobs")
      .find({ thread_id }, { projection: { _id: 0, value: 0 } })
      .sort({ channel: 1 })
      .toArray();
    expect(blobs).toEqual(
      ["list", "long", "obj"].map((channel) => ({
        thread_id,
        checkpoint_ns: "",
        channel,
        version: cp.channel_versions[channel],
        type: "json",
      }))
    );

    const tuple = await new MongoDBSaver({ client, dbName }).getTuple({
      configurable: { thread_id },
    });
    expect(tuple!.checkpoint).toEqual(cp);
  });

  it("TC-030: unchanged channels are not rewritten and still read back", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    await t.step({ a: { v: 1 }, b: { v: 1 } });
    const cp2 = await t.step({ a: { v: 2 } });
    expect(await coll("checkpoint_blobs").countDocuments({ thread_id })).toBe(3);
    const tuple = await saver.getTuple({ configurable: { thread_id } });
    expect(tuple!.checkpoint.channel_values).toEqual({ a: { v: 2 }, b: { v: 1 } });
    expect(tuple!.checkpoint).toEqual(cp2);
  });

  it("TC-034: a failed manifest write leaves no manifest; a retry succeeds", async () => {
    const failName = `checkpoints_fail_${uuid6(-1).slice(0, 8)}`;
    await client.db(dbName).createCollection(failName, {
      validator: { layout: { $exists: false } },
      validationAction: "error",
    });
    const saver = new MongoDBSaver({
      client,
      dbName,
      checkpointCollectionName: failName,
    });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    await expect(t.step({ big: { v: "x".repeat(5000) } })).rejects.toThrow();
    expect(await coll(failName).countDocuments({ thread_id })).toBe(0);
    // Blobs may have been written; they are harmless without a manifest.
    await client.db(dbName).command({ collMod: failName, validator: {} });

    // Retry the same put (same checkpoint id and versions).
    const checkpoint: Checkpoint = {
      ...emptyCheckpoint(),
      id: uuid6(-1),
      channel_values: { ...t.values },
      channel_versions: { ...t.versions },
    };
    await saver.put(
      { configurable: { thread_id, checkpoint_ns: "" } },
      checkpoint,
      meta,
      { ...t.versions }
    );
    const tuple = await saver.getTuple({ configurable: { thread_id } });
    expect(tuple!.checkpoint).toEqual(checkpoint);
  });

  it("TC-036: a missing blob raises CheckpointIntegrityError naming the pair", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    const cp = await t.step({ a: { v: 1 }, b: { v: 2 } });
    await coll("checkpoint_blobs").deleteOne({ thread_id, channel: "b" });
    const err = await saver
      .getTuple({ configurable: { thread_id } })
      .catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointIntegrityError);
    expect(err.code).toBe("CheckpointIntegrityError");
    expect(err.threadId).toBe(thread_id);
    expect(err.checkpointNs).toBe("");
    expect(err.checkpointId).toBe(cp.id);
    expect(err.missing).toEqual([
      { channel: "b", version: cp.channel_versions.b },
    ]);
    // list() must not return a partial checkpoint either.
    const listed = async () => {
      for await (const _ of saver.list({ configurable: { thread_id } })) {
        // drain
      }
    };
    await expect(listed()).rejects.toBeInstanceOf(CheckpointIntegrityError);
  });

  it("TC-001 (layout 2): a 17 MiB channel raises for checkpoint_blobs and writes nothing", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    const err = await t
      .step({ small: { a: 1 }, big: "x".repeat(17 * MiB) })
      .catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointSizeError);
    expect(err.collection).toBe("checkpoint_blobs");
    expect(err.largestChannels[0].channel).toBe("big");
    for (const c of ["checkpoints", "checkpoint_blobs"]) {
      expect(await coll(c).countDocuments({ thread_id })).toBe(0);
    }
  });

  it("TC-NF-002: 10 steps changing one 1 MB channel of five write ~1 MB of blobs per step", async () => {
    const perStep: number[] = [];
    let current = 0;
    const saver = new MongoDBSaver({
      client,
      dbName,
      onDocumentSize: (info: DocumentSizeInfo) => {
        if (info.collection === "checkpoint_blobs") current += info.bytes;
      },
    });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    const mb = (i: number) => ({ data: String(i % 10).repeat(MiB) });
    await t.step({ c0: mb(0), c1: mb(1), c2: mb(2), c3: mb(3), c4: mb(4) });
    const initial = current;
    for (let i = 0; i < 10; i += 1) {
      current = 0;
      await t.step({ [`c${i % 5}`]: mb(i + 5) });
      perStep.push(current);
    }
    expect(initial).toBeGreaterThan(5 * MiB);
    for (const bytes of perStep) {
      expect(bytes).toBeGreaterThan(MiB);
      expect(bytes).toBeLessThan(1.1 * MiB);
    }
    const total = perStep.reduce((a, b) => a + b, 0);
    expect(total).toBeLessThan(11 * MiB); // not 50 MB
    const tuple = await saver.getTuple({ configurable: { thread_id } });
    expect(tuple!.checkpoint.channel_values).toEqual(t.values);
  });

  it("pending writes come back ordered by (task_path, task_id, idx)", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const t = new ThreadDriver(saver, thread_id);
    await t.step({ a: { v: 1 } });
    await saver.putWrites(t.config, [["x", 1], ["y", 2]], "task-b");
    await saver.putWrites(t.config, [["x", 3]], "task-a");
    const tuple = await saver.getTuple(t.config);
    expect(tuple!.pendingWrites).toEqual([
      ["task-a", "x", 3],
      ["task-b", "x", 1],
      ["task-b", "y", 2],
    ]);
  });
});
