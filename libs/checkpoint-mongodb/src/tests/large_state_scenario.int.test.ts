/* eslint-disable import/no-extraneous-dependencies */
/**
 * End-to-end reproduction of the large-state situation with real StateGraphs and a
 * FRESH saver instance for every call that reads state back.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import {
  Annotation,
  Command,
  END,
  START,
  StateGraph,
  interrupt,
  messagesStateReducer,
} from "@langchain/langgraph";
import { AIMessage, HumanMessage, type BaseMessage } from "@langchain/core/messages";
import { uuid6, type BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver, CheckpointSizeError } from "../index.js";
import type { DocumentSizeInfo } from "../index.js";
import { migrateThreadLayout } from "../scripts/migrate_thread_layout.js";
import { VERSION_PATTERN } from "../versions.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";
import { loadLegacySaver } from "./legacy_saver.js";

const MiB = 1024 * 1024;

let client: MongoClient;
const dbName = uniqueDbName("large");
let LegacySaver: Awaited<ReturnType<typeof loadLegacySaver>>;

beforeAll(async () => {
  client = newClient();
  await client.connect();
  await new MongoDBSaver({ client, dbName }).setup();
  LegacySaver = await loadLegacySaver();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

const fresh = (extra: Partial<ConstructorParameters<typeof MongoDBSaver>[0]> = {}) =>
  new MongoDBSaver({ client, dbName, ...extra });
const coll = (name: string) => client.db(dbName).collection(name);

const LargeState = Annotation.Root({
  variables: Annotation<Record<string, unknown>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
  conversationHistory: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  "agentA.conversationHistory": Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  status: Annotation<string>(),
});

/** ~`total` bytes of conversation, in `count` messages. */
function conversation(total: number, count: number, tag: string): BaseMessage[] {
  const each = Math.floor(total / count);
  return Array.from({ length: count }, (_, i) =>
    i % 2 === 0
      ? new HumanMessage({ id: `${tag}-${i}`, content: `${tag}${i}:`.padEnd(each, "u") })
      : new AIMessage({ id: `${tag}-${i}`, content: `${tag}${i}:`.padEnd(each, "a") })
  );
}

function largeStateGraph(
  saver: BaseCheckpointSaver,
  sizes: { history: number; tool: number },
  toolResult?: () => string
) {
  return new StateGraph(LargeState)
    .addNode("converse", () => ({
      conversationHistory: conversation(sizes.history, 60, "c"),
      "agentA.conversationHistory": conversation(64 * 1024, 4, "a"),
      status: "conversing",
    }))
    .addNode("tool", () => ({
      variables: {
        toolResult: toolResult ? toolResult() : "t".repeat(sizes.tool),
        meta: { source: "search", rows: 3 },
      },
      status: "tool-done",
    }))
    .addNode("review", (s) => {
      const answer = interrupt({ question: "approve?", status: s.status }) as string;
      return { status: `approved:${answer}` };
    })
    .addNode("finish", (s) => ({
      status: `${s.status}|done`,
      "agentA.conversationHistory": [new AIMessage({ id: "a-final", content: "final" })],
    }))
    .addEdge(START, "converse")
    .addEdge("converse", "tool")
    .addEdge("tool", "review")
    .addEdge("review", "finish")
    .addEdge("finish", END)
    .compile({ checkpointer: saver });
}

describe("large-state scenario", () => {
  it("1. ~6 MB conversation + ~12 MB tool result: interrupt, resume from a fresh saver, all values intact", async () => {
    const sizes = { history: 6 * MiB, tool: 12 * MiB };
    const config = { configurable: { thread_id: `large1-${uuid6(-1)}` } };

    if (LegacySaver) {
      // The released 1.4.2 saver cannot store this state (one >16 MB doc).
      const legacyConfig = { configurable: { thread_id: `large1-legacy-${uuid6(-1)}` } };
      await expect(
        largeStateGraph(new LegacySaver({ client, dbName }), sizes).invoke({}, legacyConfig)
      ).rejects.toThrow(/BSON|size|large|offset is out of bounds/i);
    }

    const paused = await largeStateGraph(fresh(), sizes).invoke({}, config);
    expect((paused as { __interrupt__?: { value: unknown }[] }).__interrupt__?.[0]?.value).toEqual({
      question: "approve?",
      status: "tool-done",
    });
    const pausedState = await largeStateGraph(fresh(), sizes).getState(config);
    expect(pausedState.next).toEqual(["review"]);

    const out = await largeStateGraph(fresh(), sizes).invoke(new Command({ resume: "yes" }), config);
    expect(out.status).toBe("approved:yes|done");
    expect((out.variables.toolResult as string).length).toBe(12 * MiB);
    expect(out.variables.meta).toEqual({ source: "search", rows: 3 });
    expect(out.conversationHistory).toHaveLength(60);
    const historyBytes = out.conversationHistory.reduce(
      (n: number, m: BaseMessage) => n + (m.content as string).length,
      0
    );
    expect(historyBytes).toBeGreaterThan(5.9 * MiB);
    expect(out.conversationHistory[59].content).toBe(conversation(sizes.history, 60, "c")[59].content);
    expect(out["agentA.conversationHistory"].map((m: BaseMessage) => m.id)).toEqual([
      "a-0",
      "a-1",
      "a-2",
      "a-3",
      "a-final",
    ]);

    const state = await largeStateGraph(fresh(), sizes).getState(config);
    expect(state.values).toEqual(out);
    const manifests = await coll("checkpoints")
      .find({ thread_id: config.configurable.thread_id })
      .toArray();
    expect(manifests.every((d) => d.layout === 2)).toBe(true);
    const blobChannels = await coll("checkpoint_blobs").distinct("channel", {
      thread_id: config.configurable.thread_id,
    });
    expect(blobChannels).toEqual(
      expect.arrayContaining(["agentA.conversationHistory", "conversationHistory", "variables"])
    );
  });

  it("2. a single channel > 16 MiB raises CheckpointSizeError, writes nothing, and the thread resumes from its previous checkpoint", async () => {
    let attempt = 0;
    const toolResult = () => {
      attempt += 1;
      return attempt === 1 ? "x".repeat(17 * MiB) : "small result";
    };
    const sizes = { history: 256 * 1024, tool: 0 };
    const config = { configurable: { thread_id: `large2-${uuid6(-1)}` } };
    const thread_id = config.configurable.thread_id;

    const err = await largeStateGraph(fresh(), sizes, toolResult).invoke({}, config).catch((e) => e);
    // With the default "async" durability, invoke() rejects as soon as the
    // write fails, while the previous step's checkpoint put may still be in
    // flight: wait for it to land.
    for (let i = 0; i < 50; i += 1) {
      const s = await fresh().getTuple(config);
      if (s?.metadata?.step === 0) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(err).toBeInstanceOf(CheckpointSizeError);
    expect(err.largestChannels[0].bytes).toBeGreaterThan(17 * MiB);
    const before = await largeStateGraph(fresh(), sizes, toolResult).getState(config);
    expect(before.next).toEqual(["tool"]);
    expect(before.values.variables).toEqual({});
    // nothing of the oversized value reached the database
    for (const c of ["checkpoints", "checkpoint_writes", "checkpoint_blobs"]) {
      const big = await coll(c)
        .find({ thread_id })
        .toArray()
        .then((docs) => docs.filter((d) => JSON.stringify(d).length > 16 * MiB));
      expect(big).toHaveLength(0);
    }
    const writes = await coll("checkpoint_writes")
      .find({ thread_id, checkpoint_id: before.config.configurable!.checkpoint_id })
      .toArray();
    expect(writes.filter((w) => w.channel === "variables")).toHaveLength(0);

    // Resume from the previous checkpoint: the tool runs again and succeeds.
    const paused = await largeStateGraph(fresh(), sizes, toolResult).invoke(null, config);
    expect((paused as { __interrupt__?: { value: unknown }[] }).__interrupt__).toBeDefined();
    const out = await largeStateGraph(fresh(), sizes, toolResult).invoke(
      new Command({ resume: "ok" }),
      config
    );
    expect(out.variables.toolResult).toBe("small result");
    expect(out.status).toBe("approved:ok|done");
  });

  it("3. one channel changing per step: only the changed channel is rewritten to checkpoint_blobs", async () => {
    const State = Annotation.Root({
      a: Annotation<string>(),
      b: Annotation<string>(),
      c: Annotation<string>(),
      d: Annotation<string>(),
    });
    const value = (ch: string, step: number) => `${ch}${step}:`.padEnd(MiB, ch);
    const steps = ["a", "b", "c", "d", "a", "b", "c", "d"];
    let builder = new StateGraph(State)
      .addNode("init", () => ({ a: value("a", 0), b: value("b", 0), c: value("c", 0), d: value("d", 0) }))
      .addEdge(START, "init") as unknown as StateGraph<typeof State.spec>;
    let prev = "init";
    steps.forEach((ch, i) => {
      const name = `step${i}`;
      builder = builder.addNode(name, () => ({ [ch]: value(ch, i + 1) })) as typeof builder;
      builder = builder.addEdge(prev as never, name as never) as typeof builder;
      prev = name;
    });
    builder = builder.addEdge(prev as never, END) as typeof builder;

    const perPut: { step: number; blobBytes: number; manifestBytes: number }[] = [];
    let current = { step: NaN, blobBytes: 0, manifestBytes: 0 };
    const saver = fresh({
      onDocumentSize: (info: DocumentSizeInfo) => {
        if (info.collection === "checkpoint_blobs") current.blobBytes += info.bytes;
        if (info.collection === "checkpoints") {
          current.manifestBytes += info.bytes;
          perPut.push(current);
          current = { step: NaN, blobBytes: 0, manifestBytes: 0 };
        }
      },
    });
    const realPut = saver.put.bind(saver);
    saver.put = async (cfg, checkpoint, metadata, newVersions) => {
      current.step = metadata.step;
      return realPut(cfg, checkpoint, metadata, newVersions);
    };
    const config = { configurable: { thread_id: `large3-${uuid6(-1)}` } };
    await builder.compile({ checkpointer: saver }).invoke({}, config);

    process.stderr.write(`large-state-3 puts: ${JSON.stringify(perPut)}\n`);
    // puts: input (-1), START routing (0), init (1), step0..7 (2..9)
    const initPut = perPut.find((p) => p.step === 1)!;
    const stepPuts = perPut.filter((p) => p.step >= 2);
    expect(stepPuts).toHaveLength(steps.length);
    process.stderr.write(
      `large-state-3 bytes per put: ${JSON.stringify(perPut.map((p) => p.blobBytes + p.manifestBytes))}\n`
    );
    expect(initPut.blobBytes).toBeGreaterThan(4 * MiB);
    for (const p of stepPuts) {
      expect(p.blobBytes).toBeGreaterThan(MiB);
      expect(p.blobBytes).toBeLessThan(1.05 * MiB); // one 1 MiB channel, not 4 MiB
      expect(p.manifestBytes).toBeLessThan(8 * 1024);
    }
    const state = await builder.compile({ checkpointer: fresh() }).getState(config);
    expect(state.values).toEqual({
      a: value("a", 5),
      b: value("b", 6),
      c: value("c", 7),
      d: value("d", 8),
    });
  });

  it("4. two updateState forks of an earlier checkpoint each read their own value", async () => {
    const sizes = { history: 128 * 1024, tool: 4096 };
    const config = { configurable: { thread_id: `large4-${uuid6(-1)}` } };
    await largeStateGraph(fresh(), sizes).invoke({}, config);
    // an earlier checkpoint: right after "converse"
    let earlier;
    for await (const s of largeStateGraph(fresh(), sizes).getStateHistory(config)) {
      if (s.next[0] === "tool") earlier = s;
    }
    expect(earlier).toBeDefined();
    const left = await largeStateGraph(fresh(), sizes).updateState(earlier!.config, {
      variables: { branch: "LEFT", payload: "L".repeat(5000) },
      conversationHistory: [new HumanMessage({ id: "fork", content: "left" })],
    });
    const right = await largeStateGraph(fresh(), sizes).updateState(earlier!.config, {
      variables: { branch: "RIGHT", payload: "R".repeat(5000) },
      conversationHistory: [new HumanMessage({ id: "fork", content: "right" })],
    });
    const l = await largeStateGraph(fresh(), sizes).getState(left);
    const r = await largeStateGraph(fresh(), sizes).getState(right);
    expect(l.values.variables.branch).toBe("LEFT");
    expect(r.values.variables.branch).toBe("RIGHT");
    expect(l.values.conversationHistory.at(-1).content).toBe("left");
    expect(r.values.conversationHistory.at(-1).content).toBe("right");
    // continuing each branch keeps it isolated
    const lOut = await largeStateGraph(fresh(), sizes).invoke(null, left);
    const rOut = await largeStateGraph(fresh(), sizes).invoke(null, right);
    expect((lOut as { __interrupt__?: { value: unknown }[] }).__interrupt__).toBeDefined();
    expect((rOut as { __interrupt__?: { value: unknown }[] }).__interrupt__).toBeDefined();
    expect(lOut.variables.branch).toBe("LEFT");
    expect(rOut.variables.branch).toBe("RIGHT");
  });

  it("5. an old-format thread written by the published 1.4.2 resumes on the new code, stays old-format, and resumes identically after migration", async (ctx) => {
    if (!LegacySaver) {
      ctx.skip("LEGACY_SAVER_DIR not set");
      return;
    }
    const sizes = { history: 2 * MiB, tool: 3 * MiB };
    const a = { configurable: { thread_id: `large5-a-${uuid6(-1)}` } };
    const b = { configurable: { thread_id: `large5-b-${uuid6(-1)}` } };
    for (const config of [a, b]) {
      await largeStateGraph(new LegacySaver({ client, dbName }), sizes).invoke({}, config);
    }
    const isOldFormat = async (thread_id: string) => {
      const docs = await coll("checkpoints").find({ thread_id }).toArray();
      return docs.length > 0 && docs.every((d) => d.layout === undefined);
    };
    expect(await isOldFormat(a.configurable.thread_id)).toBe(true);

    // Resume A with the new code: works, stays integer-versioned / old format.
    const outA = await largeStateGraph(fresh(), sizes).invoke(new Command({ resume: "yes" }), a);
    expect(outA.status).toBe("approved:yes|done");
    expect(await isOldFormat(a.configurable.thread_id)).toBe(true);
    for await (const t of fresh().list(a)) {
      for (const v of Object.values(t.checkpoint.channel_versions)) {
        expect(typeof v).toBe("number");
      }
    }
    expect(await coll("checkpoint_blobs").countDocuments({ thread_id: a.configurable.thread_id })).toBe(0);

    // Migrate B (paused at the interrupt), then resume it: identical result.
    const pausedB = await largeStateGraph(fresh(), sizes).getState(b);
    const [result] = await migrateThreadLayout({
      client,
      dbName,
      threadIds: [b.configurable.thread_id],
    });
    expect(result.status).toBe("migrated");
    const migratedB = await largeStateGraph(fresh(), sizes).getState(b);
    expect(migratedB.values).toEqual(pausedB.values);
    expect(migratedB.next).toEqual(pausedB.next);
    expect(migratedB.tasks.map((t) => t.interrupts)).toEqual(pausedB.tasks.map((t) => t.interrupts));
    const docsB = await coll("checkpoints").find({ thread_id: b.configurable.thread_id }).toArray();
    expect(docsB.every((d) => d.layout === 2)).toBe(true);

    const outB = await largeStateGraph(fresh(), sizes).invoke(new Command({ resume: "yes" }), b);
    expect(outB).toEqual(outA);
    const latestB = await fresh().getTuple(b);
    for (const v of Object.values(latestB!.checkpoint.channel_versions)) {
      expect(v).toMatch(VERSION_PATTERN);
    }
    // Migrating the finished thread A also preserves its state.
    const finalA = await largeStateGraph(fresh(), sizes).getState(a);
    expect((await migrateThreadLayout({ client, dbName, threadIds: [a.configurable.thread_id] }))[0].status).toBe("migrated");
    expect((await largeStateGraph(fresh(), sizes).getState(a)).values).toEqual(finalA.values);
  });
});
