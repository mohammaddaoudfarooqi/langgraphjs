import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { CommandStartedEvent, MongoClient } from "mongodb";
import { emptyCheckpoint, uuid6 } from "@langchain/langgraph-checkpoint";
import type { Checkpoint } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../index.js";
import { dropTestDb, newClient, serverInfo, uniqueDbName } from "./int_utils.js";

const WRITE_COMMANDS = new Set(["insert", "update", "delete", "bulkWrite"]);
const meta = { source: "loop" as const, step: 1, parents: {} };

let client: MongoClient;
let maxWireVersion: number;
const dbName = uniqueDbName("write_path");
const events: CommandStartedEvent[] = [];

beforeAll(async () => {
  client = newClient({ monitorCommands: true });
  client.on("commandStarted", (e) => events.push(e));
  await client.connect();
  ({ maxWireVersion } = await serverInfo(client));
  await new MongoDBSaver({ client, dbName }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

async function recordPut(saver: MongoDBSaver, retention = false) {
  const thread_id = uuid6(-1);
  const v = saver.getNextVersion(undefined);
  const checkpoint: Checkpoint = {
    ...emptyCheckpoint(),
    id: uuid6(-1),
    channel_values: { a: { x: 1 }, b: ["y"], s: "inline" },
    channel_versions: { a: v, b: v, s: v },
  };
  // Warm up the handshake cache so it isn't counted.
  await saver.put(
    { configurable: { thread_id: `${thread_id}-warm` } },
    checkpoint,
    meta,
    checkpoint.channel_versions
  );
  events.length = 0;
  await saver.put(
    { configurable: { thread_id } },
    { ...checkpoint, id: uuid6(-1) },
    meta,
    checkpoint.channel_versions
  );
  const writes = events.filter((e) => WRITE_COMMANDS.has(e.commandName));
  const tuple = await saver.getTuple({ configurable: { thread_id } });
  expect(tuple!.checkpoint.channel_values).toEqual(checkpoint.channel_values);
  void retention;
  return writes;
}

describe("write path dispatch", () => {
  it("TC-041a (MongoDB < 8.0): collection bulkWrite for blobs, then the manifest upsert", async (ctx) => {
    if (maxWireVersion >= 25) ctx.skip();
    const writes = await recordPut(new MongoDBSaver({ client, dbName }));
    expect(writes.map((e) => [e.commandName, e.command[e.commandName]])).toEqual([
      ["update", "checkpoint_blobs"],
      ["update", "checkpoints"],
    ]);
  });

  it("TC-041b (MongoDB >= 8.0): a single client-level bulkWrite", async (ctx) => {
    if (maxWireVersion < 25) ctx.skip();
    const writes = await recordPut(new MongoDBSaver({ client, dbName }));
    expect(writes.map((e) => e.commandName)).toEqual(["bulkWrite"]);
    const ops = writes[0].command.ops as { update: number }[];
    const nsInfo = writes[0].command.nsInfo as { ns: string }[];
    // blobs first, manifest last, ordered
    expect(writes[0].command.ordered).toBe(true);
    expect(ops.map((o) => nsInfo[o.update].ns)).toEqual([
      `${dbName}.checkpoint_blobs`,
      `${dbName}.checkpoint_blobs`,
      `${dbName}.checkpoints`,
    ]);
  });

  it("REQ-NF-001: with retention, still at most 2 sequential write round trips (1 on 8.0+)", async () => {
    const saver = new MongoDBSaver({
      client,
      dbName,
      retention: { enabled: true },
    });
    await saver.setup();
    const writes = await recordPut(saver, true);
    if (maxWireVersion >= 25) {
      expect(writes.map((e) => e.commandName)).toEqual(["bulkWrite"]);
    } else {
      // blobs and threads in parallel, then the manifest
      expect(writes).toHaveLength(3);
      expect(writes[2].command.update).toBe("checkpoints");
      expect(
        writes.slice(0, 2).map((e) => e.command.update).sort()
      ).toEqual(["checkpoint_blobs", "threads"]);
    }
  });

  it("old-format (integer version) puts still issue a single update", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    events.length = 0;
    await saver.put(
      { configurable: { thread_id: uuid6(-1) } },
      {
        ...emptyCheckpoint(),
        id: uuid6(-1),
        channel_values: { a: { x: 1 } },
        channel_versions: { a: 1 },
      },
      meta,
      { a: 1 }
    );
    const writes = events.filter((e) => WRITE_COMMANDS.has(e.commandName));
    expect(writes.map((e) => [e.commandName, e.command.update])).toEqual([
      ["update", "checkpoints"],
    ]);
  });

  it("getTuple uses at most 2 sequential query steps", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const v = saver.getNextVersion(undefined);
    await saver.put(
      { configurable: { thread_id } },
      {
        ...emptyCheckpoint(),
        id: uuid6(-1),
        channel_values: { a: { x: 1 } },
        channel_versions: { a: v },
      },
      meta,
      { a: v }
    );
    events.length = 0;
    await saver.getTuple({ configurable: { thread_id } });
    const finds = events.filter((e) => e.commandName === "find");
    expect(finds.map((e) => e.command.find)).toEqual([
      "checkpoints",
      expect.any(String),
      expect.any(String),
    ]);
    expect(finds.slice(1).map((e) => e.command.find).sort()).toEqual([
      "checkpoint_blobs",
      "checkpoint_writes",
    ]);
  });
});
