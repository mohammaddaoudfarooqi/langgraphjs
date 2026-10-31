import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import type { MongoClient } from "mongodb";
import { emptyCheckpoint, uuid6 } from "@langchain/langgraph-checkpoint";
import type { Checkpoint } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver, CheckpointSizeError } from "../index.js";
import type { DocumentSizeInfo } from "../index.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

const MiB = 1024 * 1024;
const meta = { source: "update" as const, step: -1, parents: {} };

let client: MongoClient;
const dbName = uniqueDbName("size");

beforeAll(async () => {
  client = newClient();
  await client.connect();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

// Integer versions -> the single-document (old) format, so the whole state
// lands in one `checkpoints` document.
function intCheckpoint(values: Record<string, unknown>): Checkpoint {
  return {
    ...emptyCheckpoint(),
    id: uuid6(-1),
    channel_values: values,
    channel_versions: Object.fromEntries(
      Object.keys(values).map((k) => [k, 1])
    ),
  };
}

describe("size guard", () => {
  it("TC-001: put of a 17 MiB channel raises CheckpointSizeError and writes nothing", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const cp = intCheckpoint({ small: "s", big: "x".repeat(17 * MiB) });
    const err = await saver
      .put({ configurable: { thread_id } }, cp, meta, cp.channel_versions)
      .catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointSizeError);
    expect(err.code).toBe("CheckpointSizeError");
    expect(err.collection).toBe("checkpoints");
    expect(err.bytes).toBeGreaterThan(err.limit);
    expect(err.limit).toBe(16 * MiB - 16 * 1024);
    expect(err.largestChannels[0].channel).toBe("big");
    expect(err.largestChannels[0].bytes).toBeGreaterThan(17 * MiB);
    for (const coll of ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]) {
      expect(
        await client.db(dbName).collection(coll).countDocuments({ thread_id })
      ).toBe(0);
    }
  });

  it("TC-002: putWrites of a 17 MiB write raises and writes no write documents", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const config = {
      configurable: { thread_id, checkpoint_ns: "", checkpoint_id: uuid6(-1) },
    };
    const err = await saver
      .putWrites(
        config,
        [
          ["ok", "small"],
          ["big", "y".repeat(17 * MiB)],
        ],
        "task1"
      )
      .catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointSizeError);
    expect(err.collection).toBe("checkpoint_writes");
    expect(err.largestChannels[0].channel).toBe("big");
    expect(
      await client
        .db(dbName)
        .collection("checkpoint_writes")
        .countDocuments({ thread_id })
    ).toBe(0);
  });

  it("TC-003: a 13.5 MiB document succeeds, warns, and reports the error fields", async () => {
    const seen: DocumentSizeInfo[] = [];
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const saver = new MongoDBSaver({
        client,
        dbName,
        onDocumentSize: (info) => seen.push(info),
      });
      const thread_id = uuid6(-1);
      const cp = intCheckpoint({ big: "z".repeat(13.5 * MiB) });
      await saver.put(
        { configurable: { thread_id } },
        cp,
        meta,
        cp.channel_versions
      );
      expect(warn).toHaveBeenCalledTimes(1);
      expect(seen).toHaveLength(1);
      expect(seen[0].collection).toBe("checkpoints");
      expect(seen[0].bytes).toBeGreaterThan(0.8 * seen[0].limit);
      expect(seen[0].limit).toBe(16 * MiB - 16 * 1024);
      expect(seen[0].largestChannels?.[0].channel).toBe("big");
      expect(
        await client
          .db(dbName)
          .collection("checkpoints")
          .countDocuments({ thread_id })
      ).toBe(1);
    } finally {
      warn.mockRestore();
    }
  });

  it("TC-004: onDocumentSize is called once per written document", async () => {
    const seen: DocumentSizeInfo[] = [];
    const saver = new MongoDBSaver({
      client,
      dbName,
      onDocumentSize: (info) => seen.push(info),
    });
    const thread_id = uuid6(-1);
    const cp = intCheckpoint({ a: "x".repeat(1000) });
    const config = await saver.put(
      { configurable: { thread_id } },
      cp,
      meta,
      cp.channel_versions
    );
    await saver.putWrites(
      config,
      [
        ["a", "y".repeat(10)],
        ["b", "z".repeat(5000)],
      ],
      "task1"
    );
    expect(seen.map((s) => s.collection)).toEqual([
      "checkpoints",
      "checkpoint_writes",
      "checkpoint_writes",
    ]);
    const docs = await client
      .db(dbName)
      .collection("checkpoint_writes")
      .find({ thread_id })
      .sort({ idx: 1 })
      .toArray();
    const { BSON } = await import("mongodb");
    // Stored documents have an _id the saver doesn't send; sizes must match
    // what was measured before sending, so compare without _id.
    const sizes = docs.map(({ _id, ...rest }) =>
      BSON.calculateObjectSize(rest)
    );
    expect(seen.slice(1).map((s) => s.bytes)).toEqual(sizes);
  });
});
