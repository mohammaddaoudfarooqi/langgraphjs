import { describe, it, expect } from "vitest";
import { gzipSync } from "node:zlib";
import * as zlib from "node:zlib";
import type { MongoClient } from "mongodb";
import {
  CompressingSerializer,
  MongoDBSerializer,
  MongoDBSaver,
  CheckpointSerializationError,
  builtinCodecs,
} from "../index.js";

const KiB = 1024;
const big = { text: "lorem ipsum dolor sit amet ".repeat(40_000) }; // ~1 MB
const zstdAvailable = typeof (zlib as { zstdCompress?: unknown }).zstdCompress === "function";

describe("CompressingSerializer", () => {
  it("compresses values of at least minBytes and tags the type", async () => {
    const s = new CompressingSerializer(new MongoDBSerializer(), {
      codec: "gzip",
      minBytes: 64 * KiB,
    });
    const [type, bytes] = await s.dumpsTyped(big);
    expect(type).toBe("json+gzip");
    expect(bytes.length).toBeLessThan(JSON.stringify(big).length / 10);
    expect(await s.loadsTyped(type, bytes)).toEqual(big);

    const [smallType] = await s.dumpsTyped({ a: "x".repeat(10 * KiB) });
    expect(smallType).toBe("json");
  });

  it("TC-051: reads +gzip values when compression is off", async () => {
    const writer = new CompressingSerializer(new MongoDBSerializer(), {
      codec: "gzip",
      minBytes: 1,
    });
    const reader = new CompressingSerializer(new MongoDBSerializer());
    const [type, bytes] = await writer.dumpsTyped(big);
    expect(type).toBe("json+gzip");
    expect(await reader.loadsTyped(type, bytes)).toEqual(big);
    // and never compresses on write
    expect((await reader.dumpsTyped(big))[0]).toBe("json");
  });

  it("TC-052: a value whose codec isn't available raises an error naming the codec", async () => {
    const { zstd: _omit, ...withoutZstd } = builtinCodecs();
    const reader = new CompressingSerializer(new MongoDBSerializer(), {}, withoutZstd);
    const err = await reader
      .loadsTyped("json+zstd", new Uint8Array([1, 2, 3]))
      .catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointSerializationError);
    expect(err.message).toContain("zstd");
  });

  it("TC-052: configuring an unavailable codec fails at construction", () => {
    const { zstd: _omit, ...withoutZstd } = builtinCodecs();
    expect(
      () => new CompressingSerializer(new MongoDBSerializer(), { codec: "zstd" }, withoutZstd)
    ).toThrow(/zstd/);
  });

  it("TC-053: a compression bomb over maxDecompressedBytes raises", async () => {
    const bomb = gzipSync(Buffer.alloc(64 * 1024 * 1024)); // 64 MiB of zeros, ~64 KiB
    expect(bomb.length).toBeLessThan(200 * KiB);
    const reader = new CompressingSerializer(new MongoDBSerializer(), {
      maxDecompressedBytes: 1024 * 1024,
    });
    const err = await reader.loadsTyped("json+gzip", bomb).catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointSerializationError);
    expect(err.message).toMatch(/maxDecompressedBytes/);
  });

  it.skipIf(!zstdAvailable)("TC-055: zstd round trip", async () => {
    const s = new CompressingSerializer(new MongoDBSerializer(), {
      codec: "zstd",
      minBytes: 64 * KiB,
    });
    const [type, bytes] = await s.dumpsTyped(big);
    expect(type).toBe("json+zstd");
    expect(bytes.length).toBeLessThan(JSON.stringify(big).length / 10);
    expect(await s.loadsTyped(type, bytes)).toEqual(big);
  });

  it("the saver rejects an unknown codec option", () => {
    const client = { appendMetadata: () => {}, db: () => ({}) } as unknown as MongoClient;
    expect(
      () =>
        new MongoDBSaver({
          client,
          compression: { codec: "brotli" as "gzip" },
        })
    ).toThrow(/brotli/);
  });
});
