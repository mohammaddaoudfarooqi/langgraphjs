/* eslint-disable import/no-extraneous-dependencies */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
import {
  Command,
  END,
  MessagesAnnotation,
  START,
  StateGraph,
  interrupt,
  Annotation,
} from "@langchain/langgraph";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { uuid6 } from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../index.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

let client: MongoClient;
const dbName = uniqueDbName("smoke");

beforeAll(async () => {
  client = newClient();
  await client.connect();
  await new MongoDBSaver({ client, dbName }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

const TOOL_ROUNDS = 9;

const State = Annotation.Root({
  ...MessagesAnnotation.spec,
  results: Annotation<Record<string, string>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
  approved: Annotation<boolean | undefined>(),
});

/** A deterministic "model": calls a tool TOOL_ROUNDS times, then answers. */
function agent(state: typeof State.State) {
  const round = state.messages.filter((m) => m.getType() === "tool").length;
  if (round < TOOL_ROUNDS) {
    return {
      messages: [
        new AIMessage({
          content: "",
          tool_calls: [{ id: `call_${round}`, name: "lookup", args: { round } }],
        }),
      ],
    };
  }
  return { messages: [new AIMessage(`done after ${round} tool calls`)] };
}

function tools(state: typeof State.State) {
  const last = state.messages[state.messages.length - 1] as AIMessage;
  const call = last.tool_calls![0];
  // Human approval before the 5th tool call (an interrupt mid-run).
  if (call.args.round === 4 && !state.approved) {
    const answer = interrupt({ question: "approve round 4?" }) as string;
    if (answer !== "yes") throw new Error("not approved");
  }
  const output = `result-${call.args.round}:${"r".repeat(2000)}`;
  return {
    messages: [new ToolMessage({ content: output, tool_call_id: call.id! })],
    results: { [call.id!]: output.slice(0, 12) },
    approved: call.args.round >= 4 ? true : undefined,
  };
}

const graph = (saver: MongoDBSaver) =>
  new StateGraph(State)
    .addNode("agent", agent)
    .addNode("tools", tools)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", (s) => {
      const last = s.messages[s.messages.length - 1] as AIMessage;
      return last.tool_calls?.length ? "tools" : END;
    })
    .addEdge("tools", "agent")
    .compile({ checkpointer: saver });

describe("smoke", () => {
  it("TC-SMOKE-001: a 20-step agent loop with tool calls, an interrupt and a resume finishes with the expected state", async () => {
    const config = { configurable: { thread_id: uuid6(-1) }, recursionLimit: 50 };
    const first = await graph(new MongoDBSaver({ client, dbName })).invoke(
      { messages: [new HumanMessage("go")] },
      config
    );
    expect((first as { __interrupt__?: { value: unknown }[] }).__interrupt__?.[0]?.value).toEqual({ question: "approve round 4?" });

    const paused = await graph(new MongoDBSaver({ client, dbName })).getState(config);
    expect(paused.next).toEqual(["tools"]);
    expect(paused.values.messages).toHaveLength(1 + 2 * 4 + 1);

    const final = await graph(new MongoDBSaver({ client, dbName })).invoke(
      new Command({ resume: "yes" }),
      config
    );
    expect(final.messages).toHaveLength(1 + 2 * TOOL_ROUNDS + 1);
    expect(final.messages[final.messages.length - 1].content).toBe(
      `done after ${TOOL_ROUNDS} tool calls`
    );
    expect(Object.keys(final.results).sort()).toEqual(
      Array.from({ length: TOOL_ROUNDS }, (_, i) => `call_${i}`).sort()
    );
    const state = await graph(new MongoDBSaver({ client, dbName })).getState(config);
    expect(state.next).toEqual([]);
    expect(state.values).toEqual(final);

    // 20 supersteps: input + agent/tools × 9 + final agent
    let steps = 0;
    for await (const _ of new MongoDBSaver({ client, dbName }).list({
      configurable: { thread_id: config.configurable.thread_id },
    })) {
      steps += 1;
    }
    expect(steps).toBeGreaterThanOrEqual(20);
    const docs = await client
      .db(dbName)
      .collection("checkpoints")
      .find({ thread_id: config.configurable.thread_id })
      .toArray();
    expect(docs.every((d) => d.layout === 2)).toBe(true);
  });
});
