/* eslint-disable import/no-extraneous-dependencies */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { uuid6 } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../index.js";
import { VERSION_PATTERN } from "../versions.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

const MiB = 1024 * 1024;

let client: MongoClient;
const dbName = uniqueDbName("resume");

beforeAll(async () => {
  client = newClient();
  await client.connect();
  await new MongoDBSaver({ client, dbName }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

/** A fresh saver every time: nothing may survive in memory between calls. */
const freshSaver = () => new MongoDBSaver({ client, dbName });

const AB = Annotation.Root({
  a: Annotation<string | undefined>(),
  b: Annotation<string | undefined>(),
  payload: Annotation<Record<string, unknown> | undefined>(),
});

describe("resume on layout 2", () => {
  it("TC-035: a two-node graph's state reads back from a fresh saver", async () => {
    const graph = (saver: MongoDBSaver) =>
      new StateGraph(AB)
        .addNode("setA", () => ({ a: "A".repeat(3000), payload: { n: 1 } }))
        .addNode("setB", () => ({ b: "B" }))
        .addEdge(START, "setA")
        .addEdge("setA", "setB")
        .addEdge("setB", END)
        .compile({ checkpointer: saver });
    const config = { configurable: { thread_id: uuid6(-1) } };
    await graph(freshSaver()).invoke({}, config);

    const state = await graph(freshSaver()).getState(config);
    expect(state.values).toEqual({
      a: "A".repeat(3000),
      b: "B",
      payload: { n: 1 },
    });
    const manifest = await client
      .db(dbName)
      .collection("checkpoints")
      .findOne(
        { thread_id: config.configurable.thread_id },
        { sort: { checkpoint_id: -1 } }
      );
    expect(manifest!.layout).toBe(2);
    const tuple = await freshSaver().getTuple(config);
    for (const v of Object.values(tuple!.checkpoint.channel_versions)) {
      expect(v).toMatch(VERSION_PATTERN);
    }
  });

  it("TC-FORK: two updateState branches from one ancestor read back their own values", async () => {
    const graph = (saver: MongoDBSaver) =>
      new StateGraph(AB)
        .addNode("setA", () => ({ a: "root", payload: { side: "root" } }))
        .addEdge(START, "setA")
        .addEdge("setA", END)
        .compile({ checkpointer: saver });
    const thread_id = uuid6(-1);
    const config = { configurable: { thread_id } };
    await graph(freshSaver()).invoke({}, config);
    const ancestor = await graph(freshSaver()).getState(config);

    const left = await graph(freshSaver()).updateState(ancestor.config, {
      b: "LEFT",
      payload: { side: "LEFT" },
    });
    const right = await graph(freshSaver()).updateState(ancestor.config, {
      b: "RIGHT",
      payload: { side: "RIGHT" },
    });

    const l = await graph(freshSaver()).getState(left);
    const r = await graph(freshSaver()).getState(right);
    expect(l.values).toEqual({ a: "root", b: "LEFT", payload: { side: "LEFT" } });
    expect(r.values).toEqual({ a: "root", b: "RIGHT", payload: { side: "RIGHT" } });
    // Both branches started from the same versions but got distinct ones.
    const lt = await freshSaver().getTuple(left);
    const rt = await freshSaver().getTuple(right);
    expect(lt!.checkpoint.channel_versions.payload).not.toEqual(
      rt!.checkpoint.channel_versions.payload
    );
  });

  it("TC-SUBGRAPH: a subgraph's namespace state survives resume", async () => {
    const Sub = Annotation.Root({
      a: Annotation<string | undefined>(),
      inner: Annotation<Record<string, unknown> | undefined>(),
    });
    const { interrupt, Command } = await import("@langchain/langgraph");
    const sub = new StateGraph(Sub)
      .addNode("s1", () => ({ inner: { big: "i".repeat(4000) } }))
      .addNode("s2", (s) => {
        const answer = interrupt("need input");
        return { a: `${(s.inner as { big: string }).big.length}:${answer}` };
      })
      .addEdge(START, "s1")
      .addEdge("s1", "s2")
      .addEdge("s2", END)
      .compile();
    const graph = (saver: MongoDBSaver) =>
      new StateGraph(Sub)
        .addNode("sub", sub)
        .addEdge(START, "sub")
        .addEdge("sub", END)
        .compile({ checkpointer: saver });
    const config = { configurable: { thread_id: uuid6(-1) } };
    await graph(freshSaver()).invoke({}, config);
    const paused = await graph(freshSaver()).getState(config, { subgraphs: true });
    expect(paused.tasks[0].interrupts[0].value).toBe("need input");
    const subState = paused.tasks[0].state as { values: Record<string, unknown> };
    expect(subState.values.inner).toEqual({ big: "i".repeat(4000) });

    const ns = await client
      .db(dbName)
      .collection("checkpoint_blobs")
      .distinct("checkpoint_ns", { thread_id: config.configurable.thread_id });
    expect(ns.some((n) => n.startsWith("sub:"))).toBe(true);

    const out = await graph(freshSaver()).invoke(
      new Command({ resume: "yes" }),
      config
    );
    expect(out.a).toBe("4000:yes");
  });

  it("TC-BIG: a 6 MB channel plus a 12 MB channel put and cold-resume", async () => {
    const Big = Annotation.Root({
      history: Annotation<string[]>({
        reducer: (x, y) => x.concat(y),
        default: () => [],
      }),
      variables: Annotation<Record<string, unknown>>({
        reducer: (x, y) => ({ ...x, ...y }),
        default: () => ({}),
      }),
      done: Annotation<boolean | undefined>(),
    });
    const graph = (saver: MongoDBSaver) =>
      new StateGraph(Big)
        .addNode("grow", () => ({ history: ["h".repeat(6 * MiB)] }))
        .addNode("tool", () => ({ variables: { result: "r".repeat(12 * MiB) } }))
        .addNode("finish", () => ({ done: true }))
        .addEdge(START, "grow")
        .addEdge("grow", "tool")
        .addEdge("tool", "finish")
        .addEdge("finish", END)
        .compile({ checkpointer: saver, interruptBefore: ["finish"] });
    const config = { configurable: { thread_id: uuid6(-1) } };
    await graph(freshSaver()).invoke({}, config);
    const out = await graph(freshSaver()).invoke(null, config);
    expect(out.done).toBe(true);
    expect(out.history[0].length).toBe(6 * MiB);
    expect((out.variables.result as string).length).toBe(12 * MiB);
  });
});
