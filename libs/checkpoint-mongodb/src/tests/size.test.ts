import { describe, it, expect } from "vitest";
import { BSON } from "mongodb";
import { measure, checkSize, DEFAULT_MAX_DOCUMENT_BYTES } from "../size.js";
import { CheckpointSizeError } from "../errors.js";

// Small deterministic PRNG so the property test is reproducible.
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function genValue(r: () => number, depth: number): unknown {
  const kind = Math.floor(r() * (depth > 3 ? 4 : 6));
  switch (kind) {
    case 0:
      return "x".repeat(Math.floor(r() * 2000));
    case 1:
      return new Uint8Array(Math.floor(r() * 5000)).fill(7);
    case 2:
      return r() > 0.5 ? Math.floor(r() * 1e6) : r();
    case 3:
      return r() > 0.5 ? null : r() > 0.5;
    case 4: {
      const o: Record<string, unknown> = {};
      const n = Math.floor(r() * 5);
      for (let i = 0; i < n; i += 1) o[`k${i}`] = genValue(r, depth + 1);
      return o;
    }
    default: {
      const n = Math.floor(r() * 5);
      return Array.from({ length: n }, () => genValue(r, depth + 1));
    }
  }
}

describe("size", () => {
  it("TC-005: measure() equals the encoded BSON length for 20 generated documents", () => {
    const r = rng(42);
    for (let i = 0; i < 20; i += 1) {
      const doc: Record<string, unknown> = { thread_id: `t${i}` };
      for (let k = 0; k < 4; k += 1) doc[`f${k}`] = genValue(r, 0);
      expect(measure(doc)).toBe(BSON.serialize(doc).length);
      expect(measure(doc)).toBe(BSON.calculateObjectSize(doc));
    }
  });

  it("default limit is 16 MiB minus 16 KiB", () => {
    expect(DEFAULT_MAX_DOCUMENT_BYTES).toBe(16 * 1024 * 1024 - 16 * 1024);
  });

  it("checkSize throws CheckpointSizeError with the largest channels first", async () => {
    const doc = { value: new Uint8Array(2000) };
    let err: unknown;
    try {
      await checkSize(doc, {
        collection: "checkpoints",
        limit: 1000,
        warningRatio: 0.8,
        channelSizes: () => [
          { channel: "a", bytes: 10 },
          { channel: "b", bytes: 2000 },
          { channel: "c", bytes: 5 },
          { channel: "d", bytes: 6 },
          { channel: "e", bytes: 7 },
          { channel: "f", bytes: 8 },
        ],
      });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(CheckpointSizeError);
    const e = err as CheckpointSizeError;
    expect(e.code).toBe("CheckpointSizeError");
    expect(e.name).toBe("CheckpointSizeError");
    expect(e.collection).toBe("checkpoints");
    expect(e.bytes).toBeGreaterThan(1000);
    expect(e.limit).toBe(1000);
    expect(e.largestChannels.map((c) => c.channel)).toEqual([
      "b",
      "a",
      "f",
      "e",
      "d",
    ]);
  });

  it("checkSize returns the size and reports it to onDocumentSize", async () => {
    const seen: unknown[] = [];
    const bytes = await checkSize(
      { a: 1 },
      {
        collection: "checkpoint_writes",
        limit: 1000,
        warningRatio: 0.8,
        onDocumentSize: (info) => seen.push(info),
        channelSizes: () => [],
      }
    );
    expect(bytes).toBe(BSON.calculateObjectSize({ a: 1 }));
    expect(seen).toEqual([
      { collection: "checkpoint_writes", bytes, limit: 1000 },
    ]);
  });
});
