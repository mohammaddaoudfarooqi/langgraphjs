/**
 * Pure helpers for the per-channel ("layout 2") storage format. No driver or
 * serializer imports: the saver performs the I/O and serialization.
 *
 * A layout 2 checkpoint is stored as
 *
 * - one manifest document in `checkpoints`: the serialized checkpoint whose
 *   `channel_values` holds only the *inline* channels, plus the plain fields
 *   `layout: 2`, `blob_refs: [{channel, version}]` and
 *   `inline_channels: [channel]`;
 * - one document per externalized channel value in `checkpoint_blobs`, keyed
 *   by `{thread_id, checkpoint_ns, channel, version}` and written only when
 *   that channel's version changes.
 */
import type {
  ChannelVersions,
  Checkpoint,
} from "@langchain/langgraph-checkpoint";
import {
  compareWritesSortKeys,
  writesSortKey,
} from "@langchain/langgraph-checkpoint";

export const LAYOUT_VERSION = 2;

export const DEFAULT_INLINE_MAX_CHARS = 1024;

export type Placement = "inline" | "blob";

export interface BlobRef {
  channel: string;
  version: string | number;
}

/**
 * REQ-E-031: `null`, booleans, numbers and strings of at most
 * `inlineMaxChars` characters stay inline; everything else is externalized.
 * `undefined` (JS only) is treated like `null`.
 */
export function classifyChannel(
  value: unknown,
  inlineMaxChars: number = DEFAULT_INLINE_MAX_CHARS
): Placement {
  if (value === null || value === undefined) return "inline";
  switch (typeof value) {
    case "boolean":
    case "number":
      return "inline";
    case "string":
      return value.length <= inlineMaxChars ? "inline" : "blob";
    default:
      return "blob";
  }
}

export interface ManifestPlan {
  /** Copy of the checkpoint whose `channel_values` holds inline channels only. */
  checkpoint: Checkpoint;
  /** Every externalized channel that has a value, at its current version. */
  blobRefs: BlobRef[];
  /** Names of the channels stored inline. */
  inlineChannels: string[];
  /** Externalized values by channel. */
  blobValues: Record<string, unknown>;
}

/**
 * Splits a checkpoint into its manifest and externalized channels.
 *
 * A channel that has a value but no entry in `channel_versions` has no blob
 * key, so it is kept inline regardless of its type.
 */
export function buildManifest(
  checkpoint: Checkpoint,
  options: { inlineMaxChars?: number } = {}
): ManifestPlan {
  const inlineMaxChars = options.inlineMaxChars ?? DEFAULT_INLINE_MAX_CHARS;
  const inline: Record<string, unknown> = {};
  const blobRefs: BlobRef[] = [];
  const inlineChannels: string[] = [];
  const blobValues: Record<string, unknown> = {};
  const versions = checkpoint.channel_versions ?? {};
  for (const [channel, value] of Object.entries(
    checkpoint.channel_values ?? {}
  )) {
    const version = Object.prototype.hasOwnProperty.call(versions, channel)
      ? versions[channel]
      : undefined;
    if (
      version === undefined ||
      version === null ||
      classifyChannel(value, inlineMaxChars) === "inline"
    ) {
      inline[channel] = value;
      inlineChannels.push(channel);
    } else {
      blobRefs.push({ channel, version });
      blobValues[channel] = value;
    }
  }
  return {
    checkpoint: { ...checkpoint, channel_values: inline },
    blobRefs,
    inlineChannels,
    blobValues,
  };
}

export interface BlobDocDraft {
  thread_id: string;
  checkpoint_ns: string;
  channel: string;
  version: string | number;
  /** Unserialized value; the saver serializes it into `type` + `value`. */
  value: unknown;
}

/**
 * Blob documents to write for this checkpoint (REQ-E-032/033): only the
 * externalized channels named in `newVersions`, or all of them when
 * `newVersions` is undefined.
 */
export function buildBlobDocs(
  plan: ManifestPlan,
  ids: {
    threadId: string;
    checkpointNs: string;
    newVersions?: ChannelVersions;
  }
): BlobDocDraft[] {
  const { newVersions } = ids;
  return plan.blobRefs
    .filter(
      (ref) =>
        newVersions === undefined ||
        Object.prototype.hasOwnProperty.call(newVersions, ref.channel)
    )
    .map((ref) => ({
      thread_id: ids.threadId,
      checkpoint_ns: ids.checkpointNs,
      channel: ref.channel,
      version: ref.version,
      value: plan.blobValues[ref.channel],
    }));
}

/** The `$set` body of a layout 2 manifest (identity fields excluded). */
export function manifestDocument(
  plan: ManifestPlan,
  fields: {
    parentCheckpointId: string | undefined;
    type: string;
    checkpoint: Uint8Array;
    metadata: Uint8Array;
    metadataSearch: unknown;
  }
) {
  return {
    parent_checkpoint_id: fields.parentCheckpointId,
    layout: LAYOUT_VERSION,
    type: fields.type,
    checkpoint: fields.checkpoint,
    metadata: fields.metadata,
    metadata_search: fields.metadataSearch,
    blob_refs: plan.blobRefs.map((r) => ({
      channel: r.channel,
      version: r.version,
    })),
    inline_channels: plan.inlineChannels,
  };
}

/** Reassembles `channel_values` from a manifest checkpoint and its blobs. */
export function mergeBlobs(
  manifestCheckpoint: Checkpoint,
  blobs: { channel: string; value: unknown }[]
): Checkpoint {
  const channelValues: Record<string, unknown> = {
    ...manifestCheckpoint.channel_values,
  };
  for (const blob of blobs) channelValues[blob.channel] = blob.value;
  return { ...manifestCheckpoint, channel_values: channelValues };
}

export function isLayout2(doc: object): boolean {
  return (doc as { layout?: unknown }).layout === LAYOUT_VERSION;
}

/** Stable key for a `(channel, version)` pair. */
export function refKey(channel: string, version: unknown): string {
  return JSON.stringify([channel, version]);
}

/**
 * Pairs in `refs` without a matching blob (REQ-E-036). Matching is on
 * `(channel, version)` with the version's type preserved.
 */
export function missingRefs(
  refs: BlobRef[],
  found: { channel: string; version: unknown }[]
): BlobRef[] {
  const have = new Set(found.map((b) => refKey(b.channel, b.version)));
  return refs.filter((r) => !have.has(refKey(r.channel, r.version)));
}

/**
 * Pending-write order: `(task_path ?? "", task_id, idx)`, compared with the
 * shared `compareWritesSortKeys` so savers and live execution agree.
 */
export function compareWrites(
  a: { task_path?: string; task_id: string; idx: number },
  b: { task_path?: string; task_id: string; idx: number }
): number {
  return compareWritesSortKeys(
    writesSortKey(a.task_path ?? "", a.task_id, a.idx ?? 0),
    writesSortKey(b.task_path ?? "", b.task_id, b.idx ?? 0)
  );
}
