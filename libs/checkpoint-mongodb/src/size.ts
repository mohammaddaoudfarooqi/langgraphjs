import { BSON } from "mongodb";
import {
  CheckpointSizeError,
  type ChannelSize,
  type DocumentSizeInfo,
} from "./errors.js";

/** MongoDB's 16 MiB BSON limit minus 16 KiB of headroom for command overhead. */
export const DEFAULT_MAX_DOCUMENT_BYTES = 16 * 1024 * 1024 - 16 * 1024;

export const DEFAULT_SIZE_WARNING_RATIO = 0.8;

const MAX_LARGEST_CHANNELS = 5;

/** Exact BSON size of `doc`, as the driver would encode it. */
export function measure(doc: object): number {
  // The driver's default is `ignoreUndefined: false` (undefined -> null).
  return BSON.calculateObjectSize(doc as BSON.Document, {
    ignoreUndefined: false,
  });
}

export interface CheckSizeOptions {
  collection: string;
  limit: number;
  warningRatio: number;
  onDocumentSize?: (info: DocumentSizeInfo) => void;
  /**
   * Lazily computes per-channel serialized sizes. Only invoked when the
   * document is above the warning threshold, so the hot path stays cheap.
   */
  channelSizes: () => ChannelSize[] | Promise<ChannelSize[]>;
}

function largest(sizes: ChannelSize[]): ChannelSize[] {
  return [...sizes]
    .sort((a, b) => b.bytes - a.bytes)
    .slice(0, MAX_LARGEST_CHANNELS);
}

/**
 * Measures `doc` and enforces the size limit (REQ-E-001…005).
 *
 * - above `limit`: throws {@link CheckpointSizeError} (callers check every
 *   document before issuing any write, so nothing is written);
 * - above `warningRatio × limit`: `console.warn`s and reports the same fields
 *   as the error to `onDocumentSize`;
 * - otherwise: reports `{collection, bytes, limit}` to `onDocumentSize`.
 *
 * Returns the measured size.
 */
export async function checkSize(
  doc: object,
  options: CheckSizeOptions
): Promise<number> {
  const bytes = measure(doc);
  const { collection, limit, warningRatio, onDocumentSize } = options;
  if (bytes > limit) {
    throw new CheckpointSizeError({
      collection,
      bytes,
      limit,
      largestChannels: largest(await options.channelSizes()),
    });
  }
  if (bytes > warningRatio * limit) {
    const info: DocumentSizeInfo = {
      collection,
      bytes,
      limit,
      largestChannels: largest(await options.channelSizes()),
    };
    console.warn(
      `[MongoDBSaver] document for "${collection}" is ${bytes} bytes ` +
        `(${Math.round((bytes / limit) * 100)}% of the ${limit} byte limit). ` +
        `Largest channels: ${info
          .largestChannels!.map((c) => `${c.channel}=${c.bytes}`)
          .join(", ")}`
    );
    onDocumentSize?.(info);
  } else {
    onDocumentSize?.({ collection, bytes, limit });
  }
  return bytes;
}
