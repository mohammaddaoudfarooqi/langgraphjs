import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import { emptyCheckpoint, uuid6 } from "@langchain/langgraph-checkpoint";
import type { Checkpoint } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../index.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

const KiB = 1024;
let client: MongoClient;
const dbName = uniqueDbName("compression");

beforeAll(async () => {
  client = newClient();
  await client.connect();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

const oneMB = { text: "the quick brown fox jumps over the lazy dog ".repeat(24_000) };
const tenKB = { text: "k".repeat(10 * KiB) };

async function putOne(saver: MongoDBSaver, thread_id: string, metadataNote = "m") {
  const v = saver.getNextVersion(undefined);
  const checkpoint: Checkpoint = {
    ...emptyCheckpoint(),
    id: uuid6(-1),
    channel_values: { big: oneMB, small: tenKB },
    channel_versions: { big: v, small: v },
  };
  const config = await saver.put(
    { configurable: { thread_id } },
    checkpoint,
    { source: "loop", step: 1, parents: {}, note: metadataNote } as never,
    checkpoint.channel_versions
  );
  await saver.putWrites(config, [["big", oneMB], ["small", tenKB]], "task1");
  return { checkpoint, config };
}

const blobs = () => client.db(dbName).collection("checkpoint_blobs");

describe("compression", () => {
  it("TC-050: a 1 MB blob is stored gzip-compressed and smaller; a 10 KB blob is not", async () => {
    const saver = new MongoDBSaver({
      client,
      dbName,
      compression: { codec: "gzip", minBytes: 64 * KiB },
    });
    const thread_id = uuid6(-1);
    const { checkpoint, config } = await putOne(saver, thread_id);
    const big = await blobs().findOne({ thread_id, channel: "big" });
    const small = await blobs().findOne({ thread_id, channel: "small" });
    expect(big!.type).toBe("json+gzip");
    expect(big!.value.length()).toBeLessThan(JSON.stringify(oneMB).length / 10);
    expect(small!.type).toBe("json");

    const writes = await client
      .db(dbName)
      .collection("checkpoint_writes")
      .find({ thread_id })
      .sort({ idx: 1 })
      .toArray();
    expect(writes.map((w) => w.type)).toEqual(["json+gzip", "json"]);

    const tuple = await new MongoDBSaver({ client, dbName }).getTuple(config);
    expect(tuple!.checkpoint).toEqual(checkpoint);
    expect(tuple!.pendingWrites).toEqual([
      ["task1", "big", oneMB],
      ["task1", "small", tenKB],
    ]);
  });

  it("TC-051: a saver with compression off reads +gzip values", async () => {
    const thread_id = uuid6(-1);
    const { checkpoint } = await putOne(
      new MongoDBSaver({ client, dbName, compression: { minBytes: 1 } }),
      thread_id
    );
    const tuple = await new MongoDBSaver({ client, dbName }).getTuple({
      configurable: { thread_id },
    });
    expect(tuple!.checkpoint.channel_values).toEqual(checkpoint.channel_values);
  });

  it("TC-054: manifest metadata (and the manifest checkpoint) are never compressed", async () => {
    const saver = new MongoDBSaver({
      client,
      dbName,
      compression: { codec: "gzip", minBytes: 1 },
    });
    const thread_id = uuid6(-1);
    await putOne(saver, thread_id, "n".repeat(200 * KiB));
    const manifest = await client
      .db(dbName)
      .collection("checkpoints")
      .findOne({ thread_id });
    expect(manifest!.type).toBe("json");
    const metadata = JSON.parse(Buffer.from(manifest!.metadata.buffer).toString("utf8"));
    expect(metadata.note).toHaveLength(200 * KiB);
    expect(manifest!.metadata_search.note).toHaveLength(200 * KiB);
  });
});
