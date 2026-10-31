import { describe, it, expect } from "vitest";
import { emptyCheckpoint, type Checkpoint } from "@langchain/langgraph-checkpoint";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  classifyChannel,
  buildManifest,
  buildBlobDocs,
  mergeBlobs,
  isLayout2,
  manifestDocument,
  LAYOUT_VERSION,
} from "../layout.js";
import { nextVersion } from "../versions.js";

const v1 = nextVersion(undefined) as string;
const v2 = nextVersion(v1) as string;

function cp(
  values: Record<string, unknown>,
  versions?: Record<string, string>
): Checkpoint {
  return {
    ...emptyCheckpoint(),
    id: "c1",
    channel_values: values,
    channel_versions:
      versions ?? Object.fromEntries(Object.keys(values).map((k) => [k, v1])),
  };
}

describe("layout core", () => {
  it.each([
    ["null", null, "inline"],
    ["undefined", undefined, "inline"],
    ["true", true, "inline"],
    ["false", false, "inline"],
    ["integer", 42, "inline"],
    ["float", 1.5, "inline"],
    ["1024-char string", "x".repeat(1024), "inline"],
    ["1025-char string", "x".repeat(1025), "blob"],
    ["empty string", "", "inline"],
    ["object", { a: 1 }, "blob"],
    ["empty object", {}, "blob"],
    ["array", [1, 2], "blob"],
    ["empty array", [], "blob"],
    ["message", new HumanMessage("hi"), "blob"],
    ["messages", [new HumanMessage("hi"), new AIMessage("yo")], "blob"],
    ["Uint8Array", new Uint8Array([1]), "blob"],
  ])("TC-031: %s is classified %s", (_name, value, expected) => {
    expect(classifyChannel(value, 1024)).toBe(expected);
  });

  it("TC-031: inlineMaxChars is configurable", () => {
    expect(classifyChannel("abc", 2)).toBe("blob");
    expect(classifyChannel("ab", 2)).toBe("inline");
  });

  it("buildManifest splits inline and blob channels", () => {
    const x = cp(
      { s: "short", n: 1, obj: { a: 1 }, long: "y".repeat(2000) },
      { s: v1, n: v1, obj: v2, long: v1, empty: v1 }
    );
    const plan = buildManifest(x, { inlineMaxChars: 1024 });
    expect(plan.inlineChannels.sort()).toEqual(["n", "s"]);
    expect(plan.blobRefs).toEqual([
      { channel: "obj", version: v2 },
      { channel: "long", version: v1 },
    ]);
    expect(plan.checkpoint.channel_values).toEqual({ s: "short", n: 1 });
    // channel_versions etc. are kept unchanged on the manifest
    expect(plan.checkpoint.channel_versions).toEqual(x.channel_versions);
    // The input checkpoint is not mutated.
    expect(Object.keys(x.channel_values)).toHaveLength(4);
  });

  it("a value without a channel version stays inline (no blob key exists)", () => {
    const x = cp({ obj: { a: 1 } }, {});
    const plan = buildManifest(x, { inlineMaxChars: 1024 });
    expect(plan.blobRefs).toEqual([]);
    expect(plan.inlineChannels).toEqual(["obj"]);
  });

  it("TC-032: buildBlobDocs only covers channels in newVersions", () => {
    const x = cp({ a: { a: 1 }, b: { b: 1 }, c: "inline" });
    const plan = buildManifest(x, { inlineMaxChars: 1024 });
    const docs = buildBlobDocs(plan, {
      threadId: "t",
      checkpointNs: "",
      newVersions: { a: v1, c: v1 },
    });
    expect(docs).toEqual([
      { thread_id: "t", checkpoint_ns: "", channel: "a", version: v1, value: { a: 1 } },
    ]);
  });

  it("TC-033: without newVersions every blob channel is covered", () => {
    const x = cp({ a: { a: 1 }, b: { b: 1 }, c: "inline" });
    const plan = buildManifest(x, { inlineMaxChars: 1024 });
    const docs = buildBlobDocs(plan, { threadId: "t", checkpointNs: "ns" });
    expect(docs.map((d) => d.channel)).toEqual(["a", "b"]);
    expect(docs.every((d) => d.checkpoint_ns === "ns")).toBe(true);
  });

  it("property: mergeBlobs(buildManifest(x), blobs(x)) reproduces x.channel_values", () => {
    let seed = 7;
    const r = () => {
      seed = (seed * 1103515245 + 12345) % 2 ** 31;
      return seed / 2 ** 31;
    };
    const gen = (): unknown => {
      const k = Math.floor(r() * 7);
      if (k === 0) return null;
      if (k === 1) return r() > 0.5;
      if (k === 2) return Math.floor(r() * 100);
      if (k === 3) return "s".repeat(Math.floor(r() * 2048));
      if (k === 4) return { n: r() };
      if (k === 5) return [r(), "a"];
      return undefined;
    };
    for (let i = 0; i < 100; i += 1) {
      const values: Record<string, unknown> = {};
      const n = Math.floor(r() * 8);
      for (let c = 0; c < n; c += 1) values[`ch${c}`] = gen();
      const x = cp(values);
      const plan = buildManifest(x, { inlineMaxChars: 1024 });
      const blobs = buildBlobDocs(plan, { threadId: "t", checkpointNs: "" });
      const merged = mergeBlobs(
        plan.checkpoint,
        blobs.map((b) => ({ channel: b.channel, value: b.value }))
      );
      expect(merged.channel_values).toEqual(x.channel_values);
      expect(Object.keys(merged.channel_values).sort()).toEqual(
        Object.keys(x.channel_values).sort()
      );
    }
  });

  it("manifestDocument has the layout 2 shape", () => {
    const x = cp({ a: { a: 1 }, s: "s" });
    const plan = buildManifest(x, { inlineMaxChars: 1024 });
    const doc = manifestDocument(plan, {
      parentCheckpointId: "p",
      type: "json",
      checkpoint: new Uint8Array([1]),
      metadata: new Uint8Array([2]),
      metadataSearch: { step: 1 },
    });
    expect(doc).toEqual({
      parent_checkpoint_id: "p",
      layout: LAYOUT_VERSION,
      type: "json",
      checkpoint: new Uint8Array([1]),
      metadata: new Uint8Array([2]),
      metadata_search: { step: 1 },
      blob_refs: [{ channel: "a", version: v1 }],
      inline_channels: ["s"],
    });
    expect(isLayout2(doc)).toBe(true);
    expect(isLayout2({ type: "json" })).toBe(false);
  });
});
