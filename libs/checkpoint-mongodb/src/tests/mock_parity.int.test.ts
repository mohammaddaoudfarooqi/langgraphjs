import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import { DeltaSnapshot, emptyCheckpoint } from "@langchain/langgraph-checkpoint";
import type { Checkpoint, CheckpointTuple } from "@langchain/langgraph-checkpoint";
import { HumanMessage } from "@langchain/core/messages";
import { MongoDBSaver } from "../index.js";
import { createInMemoryClient } from "./mock_client.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

let client: MongoClient;
const dbName = uniqueDbName("parity");

beforeAll(async () => {
  client = newClient();
  await client.connect();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

const S = (n: number) => `${String(n).padStart(32, "0")}.0.${String(n).padStart(14, "7")}`;

/** A fixed sequence of saver calls; returns every observable result. */
async function scenario(saver: MongoDBSaver) {
  const out: unknown[] = [];
  const meta = (step: number) => ({ source: "loop" as const, step, parents: {}, tag: `t${step}` });
  const cp = (id: string, values: Record<string, unknown>, versions: Record<string, string | number>): Checkpoint => ({
    ...emptyCheckpoint(),
    id,
    ts: "2026-10-08T00:00:00.000Z",
    channel_values: values,
    channel_versions: versions,
  });
  const strip = (t: CheckpointTuple | undefined) => t && { ...t };

  // layout 2 thread, one channel changing per step
  let config = await saver.put(
    { configurable: { thread_id: "p1", checkpoint_ns: "" } },
    cp("1ef0000000000001", { msgs: [new HumanMessage("hi")], s: "short", big: "b".repeat(3000) }, { msgs: S(1), s: S(1), big: S(1) }),
    meta(0),
    { msgs: S(1), s: S(1), big: S(1) }
  );
  out.push(config);
  await saver.putWrites(config, [["msgs", [new HumanMessage("w1")]], ["s", "x"]], "task-b");
  await saver.putWrites(config, [["msgs", ["w0"]]], "task-a");
  await saver.putWrites(config, [["__interrupt__", { v: 1 }]], "task-a");
  config = await saver.put(
    config,
    cp("1ef0000000000002", { msgs: new DeltaSnapshot(["x"]), s: "short", big: "b".repeat(3000) }, { msgs: S(2), s: S(1), big: S(1) }),
    meta(1),
    { msgs: S(2) }
  );
  out.push(config);
  // old-format thread with integer versions
  const old = await saver.put(
    { configurable: { thread_id: "p2", checkpoint_ns: "ns" } },
    cp("1ef0000000000003", { a: { x: 1 } }, { a: 1 }),
    meta(0),
    { a: 1 }
  );
  await saver.putWrites(old, [["a", 2]], "t");

  out.push(strip(await saver.getTuple({ configurable: { thread_id: "p1" } })));
  out.push(strip(await saver.getTuple({ configurable: { thread_id: "p1", checkpoint_id: "1ef0000000000001" } })));
  out.push(strip(await saver.getTuple({ configurable: { thread_id: "p2", checkpoint_ns: "ns" } })));
  out.push(await saver.getTuple({ configurable: { thread_id: "missing" } }));
  const listed: unknown[] = [];
  for await (const t of saver.list({ configurable: { thread_id: "p1" } }, { filter: { tag: "t0" } })) listed.push(t);
  for await (const t of saver.list({}, { limit: 2 })) listed.push(t);
  for await (const t of saver.list({ configurable: { thread_id: "p1" } }, { before: { configurable: { checkpoint_id: "1ef0000000000002" } } })) listed.push(t);
  out.push(listed);
  out.push(await saver.getDeltaChannelHistory({ config, channels: ["msgs", "s", "nope"] }));
  await saver.deleteThread("p2");
  out.push(await saver.getTuple({ configurable: { thread_id: "p2", checkpoint_ns: "ns" } }));
  return out;
}

describe("mock parity", () => {
  it("TC-PARITY-001: the in-memory mock client returns the same results as the real server", async () => {
    const real = await scenario(new MongoDBSaver({ client, dbName }));
    const mocked = await scenario(new MongoDBSaver({ client: createInMemoryClient(), dbName }));
    const mocked8 = await scenario(new MongoDBSaver({ client: createInMemoryClient(25), dbName }));
    expect(mocked).toEqual(real);
    expect(mocked8).toEqual(real);
    expect(real.length).toBe(9);
    const latest = real[2] as CheckpointTuple;
    expect(latest.checkpoint.channel_values.big).toBe("b".repeat(3000));
    expect(latest.pendingWrites).toEqual([]);
    const first = real[3] as CheckpointTuple;
    expect(first.pendingWrites!.map((w) => w[0])).toEqual(["task-a", "task-a", "task-b", "task-b"]);
    expect(real[5]).toBeUndefined();
    expect((real[6] as unknown[]).length).toBe(4);
  });
});
