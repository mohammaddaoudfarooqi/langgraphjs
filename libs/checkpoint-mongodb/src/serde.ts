/* eslint-disable @typescript-eslint/no-explicit-any */
import {
  MemorySaver,
  type SerializerProtocol,
} from "@langchain/langgraph-checkpoint";
import * as zlib from "node:zlib";
import { CheckpointSerializationError } from "./errors.js";

/**
 * The upstream safe stringifier catches `JSON.stringify` failures and returns
 * this string (JSON-encoded) instead of throwing.
 */
const PLACEHOLDER =
  "[unable to serialize, circular reference is too complex to analyze]";
const PLACEHOLDER_JSON = JSON.stringify(PLACEHOLDER);
const PLACEHOLDER_BYTES = new TextEncoder().encode(PLACEHOLDER_JSON);

function isPlaceholder(bytes: Uint8Array): boolean {
  if (bytes.length !== PLACEHOLDER_BYTES.length) return false;
  for (let i = 0; i < bytes.length; i += 1) {
    if (bytes[i] !== PLACEHOLDER_BYTES[i]) return false;
  }
  return true;
}

/**
 * A fresh instance of the upstream `JsonPlusSerializer`. The class itself is
 * not exported by `@langchain/langgraph-checkpoint`, but every saver is
 * constructed with one as its default `serde`.
 */
export function createJsonPlusSerializer(): SerializerProtocol {
  return new MemorySaver().serde;
}

/**
 * Default serializer of the MongoDB saver (REQ-E-010).
 *
 * Wraps the upstream `JsonPlusSerializer` and produces identical output,
 * except that values the safe stringifier cannot encode raise
 * {@link CheckpointSerializationError} rather than being stored as a
 * placeholder string — which would silently lose the state on resume.
 */
export class MongoDBSerializer implements SerializerProtocol {
  protected inner: SerializerProtocol = createJsonPlusSerializer();

  async dumpsTyped(obj: any): Promise<[string, Uint8Array]> {
    const [type, bytes] = await this.inner.dumpsTyped(obj);
    // The placeholder replaces the *whole* output, so a byte-exact match means
    // serialization failed (unless the value really is that exact string).
    if (type === "json" && isPlaceholder(bytes) && obj !== PLACEHOLDER) {
      let cause: unknown;
      try {
        JSON.stringify(obj);
      } catch (e) {
        cause = e;
      }
      throw new CheckpointSerializationError(
        `Failed to serialize checkpoint value: ${
          (cause as Error | undefined)?.message ?? "value is not serializable"
        }`,
        { cause }
      );
    }
    return [type, bytes];
  }

  async loadsTyped(type: string, data: Uint8Array | string): Promise<any> {
    return this.inner.loadsTyped(type, data);
  }
}

// ---------------------------------------------------------------------------
// Compression (REQ-E-050…055)
// ---------------------------------------------------------------------------

export type CompressionCodecName = "gzip" | "zstd";

export interface CompressionOptions {
  /** Codec used for new values. Defaults to `"gzip"`. */
  codec?: CompressionCodecName;
  /** Values smaller than this (serialized) are stored uncompressed. Default 64 KiB. */
  minBytes?: number;
  /** Decompression stops and raises past this size. Default 64 × 16 MiB. */
  maxDecompressedBytes?: number;
}

export const DEFAULT_COMPRESSION_MIN_BYTES = 64 * 1024;

export const DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 16 * 1024 * 1024;

export interface CompressionCodec {
  compress(data: Uint8Array): Promise<Uint8Array>;
  /** Must reject once the output would exceed `maxOutputLength`. */
  decompress(data: Uint8Array, maxOutputLength: number): Promise<Uint8Array>;
}

type ZlibFn = (
  buf: Uint8Array,
  options: Record<string, unknown>,
  cb: (err: Error | null, result: Buffer) => void
) => void;

function promisifyZlib(fn: ZlibFn) {
  return (buf: Uint8Array, options: Record<string, unknown> = {}) =>
    new Promise<Uint8Array>((resolve, reject) => {
      fn(buf, options, (err, result) => (err ? reject(err) : resolve(result)));
    });
}

/**
 * Codecs available in this runtime: `gzip` always (built-in `zlib`), `zstd`
 * where Node provides `zlib.zstdCompress` (Node 22.15+ / 23.8+).
 */
export function builtinCodecs(): Record<string, CompressionCodec> {
  const codecs: Record<string, CompressionCodec> = {
    gzip: {
      compress: (d) => promisifyZlib(zlib.gzip as unknown as ZlibFn)(d),
      decompress: (d, max) =>
        promisifyZlib(zlib.gunzip as unknown as ZlibFn)(d, {
          maxOutputLength: max,
        }),
    },
  };
  const z = zlib as unknown as {
    zstdCompress?: ZlibFn;
    zstdDecompress?: ZlibFn;
  };
  if (
    typeof z.zstdCompress === "function" &&
    typeof z.zstdDecompress === "function"
  ) {
    codecs.zstd = {
      compress: (d) => promisifyZlib(z.zstdCompress!)(d),
      decompress: (d, max) =>
        promisifyZlib(z.zstdDecompress!)(d, { maxOutputLength: max }),
    };
  }
  return codecs;
}

/**
 * Wraps a serializer and compresses large values as `<type>+<codec>`.
 *
 * Reading always understands `+gzip` / `+zstd` types, whether or not
 * compression is configured for writing (REQ-E-051). A stored codec that this
 * process cannot decode raises an error naming it (REQ-E-052); output larger
 * than `maxDecompressedBytes` raises (REQ-E-053).
 */
export class CompressingSerializer implements SerializerProtocol {
  readonly codec?: CompressionCodecName;

  readonly minBytes: number;

  readonly maxDecompressedBytes: number;

  constructor(
    readonly inner: SerializerProtocol,
    options?: CompressionOptions,
    protected codecs: Record<string, CompressionCodec> = builtinCodecs()
  ) {
    this.minBytes = options?.minBytes ?? DEFAULT_COMPRESSION_MIN_BYTES;
    this.maxDecompressedBytes =
      options?.maxDecompressedBytes ?? DEFAULT_MAX_DECOMPRESSED_BYTES;
    if (options?.codec !== undefined) {
      if (this.codecs[options.codec] === undefined) {
        throw new CheckpointSerializationError(
          `Compression codec "${options.codec}" is not available in this ` +
            `runtime (available: ${Object.keys(this.codecs).join(", ")}).` +
            (options.codec === "zstd"
              ? " zstd requires Node.js 22.15+ (zlib.zstdCompress)."
              : "")
        );
      }
      this.codec = options.codec;
    }
  }

  async dumpsTyped(obj: any): Promise<[string, Uint8Array]> {
    const [type, bytes] = await this.inner.dumpsTyped(obj);
    if (this.codec === undefined || bytes.length < this.minBytes) {
      return [type, bytes];
    }
    const compressed = await this.codecs[this.codec].compress(bytes);
    // Incompressible data is stored as-is.
    if (compressed.length >= bytes.length) return [type, bytes];
    return [`${type}+${this.codec}`, compressed];
  }

  async loadsTyped(type: string, data: Uint8Array | string): Promise<any> {
    const plus = type.lastIndexOf("+");
    if (plus < 0) return this.inner.loadsTyped(type, data);
    const baseType = type.slice(0, plus);
    const codecName = type.slice(plus + 1);
    const codec = this.codecs[codecName];
    if (codec === undefined) {
      throw new CheckpointSerializationError(
        `Cannot decode a value stored with compression codec "${codecName}": ` +
          `it is not available in this runtime` +
          (codecName === "zstd"
            ? " (zstd requires Node.js 22.15+ with zlib.zstdDecompress)."
            : ".")
      );
    }
    const bytes =
      typeof data === "string" ? new TextEncoder().encode(data) : data;
    let decompressed: Uint8Array;
    try {
      decompressed = await codec.decompress(bytes, this.maxDecompressedBytes);
    } catch (e) {
      const tooLarge =
        (e as Error)?.name === "RangeError" ||
        (e as { code?: string }).code === "ERR_BUFFER_TOO_LARGE";
      throw new CheckpointSerializationError(
        tooLarge
          ? `Decompressed ${codecName} value exceeds maxDecompressedBytes ` +
              `(${this.maxDecompressedBytes} bytes)`
          : `Failed to decompress ${codecName} value: ${(e as Error).message}`,
        { cause: e }
      );
    }
    return this.inner.loadsTyped(baseType, decompressed);
  }
}
