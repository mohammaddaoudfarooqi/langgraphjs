/* eslint-disable import/no-extraneous-dependencies */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import {
  Annotation,
  Command,
  END,
  START,
  StateGraph,
  interrupt,
} from "@langchain/langgraph";
import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import { uuid6 } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../index.js";
import { VERSION_PATTERN } from "../versions.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";
import { loadLegacySaver } from "./legacy_saver.js";

let client: MongoClient;
const dbName = uniqueDbName("mixed");
let LegacySaver: Awaited<ReturnType<typeof loadLegacySaver>>;

beforeAll(async () => {
  client = newClient();
  await client.connect();
  LegacySaver = await loadLegacySaver();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

const State = Annotation.Root({
  log: Annotation<string[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  data: Annotation<Record<string, unknown>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
  answer: Annotation<string | undefined>(),
});

const graph = (saver: BaseCheckpointSaver) =>
  new StateGraph(State)
    .addNode("collect", () => ({ log: ["collect"], data: { big: "d".repeat(5000) } }))
    .addNode("ask", () => ({ answer: interrupt("question?") as string, log: ["ask"] }))
    .addNode("finish", (s) => ({ log: [`finish:${s.answer}`] }))
    .addEdge(START, "collect")
    .addEdge("collect", "ask")
    .addEdge("ask", "finish")
    .addEdge("finish", END)
    .compile({ checkpointer: saver });

const fresh = () => new MongoDBSaver({ client, dbName });

describe("mixed old-format and layout 2 threads", () => {
  it("TC-039 / TC-UPGRADE: an old-format thread and a new thread both resume; the old one stays integer-versioned and old-format", async (ctx) => {
    if (!LegacySaver) {
      ctx.skip("LEGACY_SAVER_DIR not set (npm install @langchain/langgraph-checkpoint-mongodb@1.4.2)");
      return;
    }
    const oldThread = { configurable: { thread_id: `old-${uuid6(-1)}` } };
    const newThread = { configurable: { thread_id: `new-${uuid6(-1)}` } };

    // Written by the released 1.4.2 package, paused at the interrupt.
    await graph(new LegacySaver!({ client, dbName })).invoke({ log: ["start"] }, oldThread);
    const legacyView = await new LegacySaver!({ client, dbName }).getTuple(oldThread);
    // Written by the new code.
    await graph(fresh()).invoke({ log: ["start"] }, newThread);

    // INV-001: the new reader returns exactly what the old reader returns.
    const newView = await fresh().getTuple(oldThread);
    expect(newView!.checkpoint).toEqual(legacyView!.checkpoint);
    expect(newView!.metadata).toEqual(legacyView!.metadata);
    expect(newView!.pendingWrites).toEqual(
      [...legacyView!.pendingWrites!].sort((a, b) =>
        a[0] === b[0] ? 0 : a[0] < b[0] ? -1 : 1
      )
    );

    const oldOut = await graph(fresh()).invoke(new Command({ resume: "yes" }), oldThread);
    const newOut = await graph(fresh()).invoke(new Command({ resume: "yes" }), newThread);
    expect(oldOut.log).toEqual(["start", "collect", "ask", "finish:yes"]);
    expect(newOut.log).toEqual(oldOut.log);
    expect(oldOut.data).toEqual(newOut.data);

    const coll = client.db(dbName).collection("checkpoints");
    // INV-002: every checkpoint of the old thread is old-format with integer versions.
    const oldDocs = await coll.find({ thread_id: oldThread.configurable.thread_id }).toArray();
    expect(oldDocs.length).toBeGreaterThan(3);
    expect(oldDocs.every((d) => d.layout === undefined && d.blob_refs === undefined)).toBe(true);
    for await (const t of fresh().list(oldThread)) {
      for (const v of Object.values(t.checkpoint.channel_versions)) {
        expect(typeof v).toBe("number");
      }
    }
    expect(
      await client
        .db(dbName)
        .collection("checkpoint_blobs")
        .countDocuments({ thread_id: oldThread.configurable.thread_id })
    ).toBe(0);

    // The new thread is layout 2 with string versions throughout.
    const newDocs = await coll.find({ thread_id: newThread.configurable.thread_id }).toArray();
    expect(newDocs.every((d) => d.layout === 2)).toBe(true);
    const latest = await fresh().getTuple(newThread);
    for (const v of Object.values(latest!.checkpoint.channel_versions)) {
      expect(v).toMatch(VERSION_PATTERN);
    }

    // The released package can still read the old thread after the new code
    // wrote to it (rollback safety).
    const back = await new LegacySaver!({ client, dbName }).getTuple(oldThread);
    expect(back!.checkpoint.channel_values.log).toEqual(oldOut.log);
  });
});
