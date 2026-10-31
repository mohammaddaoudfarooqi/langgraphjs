import { describe, it, expect, vi } from "vitest";
import type { MongoClient } from "mongodb";
import { emptyCheckpoint, uuid6 } from "@langchain/langgraph-checkpoint";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  MongoDBSaver,
  MongoDBSerializer,
  CheckpointSerializationError,
} from "../index.js";
import { createJsonPlusSerializer } from "../serde.js";

// The upstream JsonPlusSerializer (not exported by name).
const JsonPlusSerializer = function JsonPlusSerializer() {
  return createJsonPlusSerializer();
} as unknown as { new (): ReturnType<typeof createJsonPlusSerializer> };

const PLACEHOLDER = "[unable to serialize";

function recordingClient() {
  const writes: unknown[] = [];
  const collection = {
    updateOne: vi.fn(async (...args: unknown[]) => {
      writes.push(args);
      return { acknowledged: true };
    }),
    bulkWrite: vi.fn(async (...args: unknown[]) => {
      writes.push(args);
      return { acknowledged: true };
    }),
  };
  const client = {
    appendMetadata: vi.fn(),
    bulkWrite: vi.fn(async (...args: unknown[]) => {
      writes.push(args);
      return {};
    }),
    db: vi.fn(() => ({
      collection: vi.fn(() => collection),
      command: vi.fn(async () => ({ maxWireVersion: 21, setName: "rs" })),
    })),
  };
  return { client: client as unknown as MongoClient, writes };
}

// A circular structure that also holds a value JSON.stringify rejects: the
// safe stringifier strips the cycle, then JSON.stringify throws and the
// upstream serializer returns a placeholder string instead of failing.
function tooComplex() {
  const a: Record<string, unknown> = { n: 10n };
  a.self = a;
  return a;
}

const throwingToJSON = {
  toJSON() {
    throw new Error("boom");
  },
};

describe("MongoDBSerializer", () => {
  it("the upstream serializer silently stores a placeholder (motivation)", async () => {
    const [, bytes] = await new JsonPlusSerializer().dumpsTyped(tooComplex());
    expect(new TextDecoder().decode(bytes)).toContain(PLACEHOLDER);
  });

  it.each([
    ["circular + bigint", tooComplex()],
    ["throwing toJSON", throwingToJSON],
  ])("TC-010: dumpsTyped throws for %s", async (_name, value) => {
    await expect(
      new MongoDBSerializer().dumpsTyped({ v: value })
    ).rejects.toBeInstanceOf(CheckpointSerializationError);
  });

  it.each([
    ["integer versions", 1],
    ["string versions", "00000000000000000000000000000001.0.12345678901234"],
  ])(
    "TC-010: put throws with %s and nothing containing the placeholder is written",
    async (_name, version) => {
      const { client, writes } = recordingClient();
      const saver = new MongoDBSaver({ client });
      const checkpoint = {
        ...emptyCheckpoint(),
        id: uuid6(-1),
        channel_values: { bad: tooComplex() },
        channel_versions: { bad: version },
      };
      await expect(
        saver.put({ configurable: { thread_id: "t" } }, checkpoint, {
          source: "loop",
          step: 1,
          parents: {},
        }, { bad: version })
      ).rejects.toBeInstanceOf(CheckpointSerializationError);
      const stored = JSON.stringify(writes, (_k, v) =>
        v instanceof Uint8Array ? new TextDecoder().decode(v) : v
      );
      expect(stored).not.toContain(PLACEHOLDER);
    }
  );

  it("TC-010: putWrites throws instead of storing a placeholder", async () => {
    const { client, writes } = recordingClient();
    const saver = new MongoDBSaver({ client });
    await expect(
      saver.putWrites(
        {
          configurable: {
            thread_id: "t",
            checkpoint_ns: "",
            checkpoint_id: uuid6(-1),
          },
        },
        [["bad", tooComplex()]],
        "task"
      )
    ).rejects.toBeInstanceOf(CheckpointSerializationError);
    expect(writes).toHaveLength(0);
  });

  it("TC-011: a custom serde is used unchanged", async () => {
    const { client } = recordingClient();
    const custom = new JsonPlusSerializer();
    const spy = vi.spyOn(custom, "dumpsTyped");
    const defaultSpy = vi.spyOn(MongoDBSerializer.prototype, "dumpsTyped");
    try {
      const saver = new MongoDBSaver({ client }, custom);
      const checkpoint = {
        ...emptyCheckpoint(),
        id: uuid6(-1),
        channel_values: { a: { x: 1 } },
        channel_versions: { a: 1 },
      };
      await saver.put(
        { configurable: { thread_id: "t" } },
        checkpoint,
        { source: "loop", step: 1, parents: {} },
        { a: 1 }
      );
      expect(spy).toHaveBeenCalled();
      expect(defaultSpy).not.toHaveBeenCalled();
    } finally {
      defaultSpy.mockRestore();
    }
  });

  it("TC-011: the default serde is a MongoDBSerializer", () => {
    const { client } = recordingClient();
    const saver = new MongoDBSaver({ client });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((saver as any).baseSerde).toBeInstanceOf(MongoDBSerializer);
  });

  it("TC-013: round trips match JsonPlusSerializer", async () => {
    const value = {
      messages: [new HumanMessage("hi"), new AIMessage("hello")],
      map: new Map<string, unknown>([["a", 1], ["b", { c: 2 }]]),
      set: new Set([1, 2, 3]),
      bytes: new Uint8Array([1, 2, 3]),
      date: new Date("2024-01-02T03:04:05.000Z"),
      nested: { u: undefined, n: null, s: "x" },
    };
    const ours = new MongoDBSerializer();
    const theirs = new JsonPlusSerializer();
    const [t1, b1] = await ours.dumpsTyped(value);
    const [t2, b2] = await theirs.dumpsTyped(value);
    expect(t1).toBe(t2);
    expect(new TextDecoder().decode(b1)).toBe(new TextDecoder().decode(b2));
    expect(await ours.loadsTyped(t1, b1)).toEqual(
      await theirs.loadsTyped(t2, b2)
    );
    const raw = new Uint8Array([9, 8, 7]);
    expect(await ours.dumpsTyped(raw)).toEqual(await theirs.dumpsTyped(raw));
  });
});
