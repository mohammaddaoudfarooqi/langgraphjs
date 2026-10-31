import type { ChannelVersions } from "@langchain/langgraph-checkpoint";

/**
 * Channel version format for layout 2 threads (REQ-E-037):
 * a 32-digit zero-padded integer prefix, a dot, and a 16-character random
 * fraction, e.g. `"00000000000000000000000000000007.0.83751409126543"`.
 *
 * Ordering is lexicographic on the prefix. The random suffix makes versions
 * produced by sibling forks distinct, so per-channel blobs keyed by
 * `(channel, version)` never collide (premortem #2).
 */
export const VERSION_PATTERN = /^\d{32}\.[0-9.]{16}$/;

const PREFIX_DIGITS = 32;

/**
 * Random 16-character suffix: `"0."` followed by 14 digits. `toFixed(14)`
 * always yields exactly 16 characters with no exponent (Python equivalent:
 * `f"{random.random():.14f}"`).
 */
function randomSuffix(): string {
  let r = Math.random().toFixed(14);
  // toFixed can round 0.999999999999995+ up to "1.00000000000000".
  if (!r.startsWith("0.")) r = "0.99999999999999";
  return r;
}

export function formatVersion(n: number, suffix = randomSuffix()): string {
  return `${String(n).padStart(PREFIX_DIGITS, "0")}.${suffix}`;
}

/** Integer prefix of a string version (`"…0005.x"` -> 5). */
export function versionPrefix(version: string): number {
  const n = Number.parseInt(version.split(".")[0], 10);
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid channel version: ${JSON.stringify(version)}`);
  }
  return n;
}

/**
 * Next channel version (REQ-E-037/038).
 *
 * - `undefined` or a string -> string version with the prefix incremented;
 * - a number -> that number plus one, so threads written with integer
 *   versions (released 1.4.x) keep integer versions and the old format.
 */
export function nextVersion(
  current: number | string | undefined
): number | string {
  if (typeof current === "number") return current + 1;
  const n = current === undefined ? 0 : versionPrefix(current);
  return formatVersion(n + 1);
}

export type VersionKind = "string" | "number" | "empty";

/**
 * `"string"` if any version is a string, `"number"` if all are numbers,
 * `"empty"` if there are none. Strings and empty use layout 2.
 */
export function versionKind(channelVersions: ChannelVersions): VersionKind {
  let kind: VersionKind = "empty";
  for (const v of Object.values(channelVersions ?? {})) {
    if (typeof v === "string") return "string";
    kind = "number";
  }
  return kind;
}
