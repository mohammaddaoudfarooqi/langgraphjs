/**
 * Layout benchmark (REQ-NF-003): old single-document format vs per-channel
 * layout 2, at 1 MB, 6 MB and 15 MB of total state, with one channel
 * changing per step.
 *
 *   MONGODB_URL=... npx tsx bench/layout.ts [--iterations 50]
 *
 * Uses (and drops) a database named `ckpt_test_js_bench_<random>`.
 * Reports p50/p99 latency of put, putWrites and getTuple, plus the bytes
 * written per step (BSON size of every document sent).
 */
import { randomBytes } from "node:crypto";
import { performance } from "node:perf_hooks";
import { MongoClient } from "mongodb";
import {
  emptyCheckpoint,
  uuid6,
  type ChannelVersions,
  type Checkpoint,
} from "@langchain/langgraph-checkpoint";
import { MongoDBSaver } from "../src/index.js";

const MB = 1_000_000;
const CHANNELS = 5;

class OldFormatSaver extends MongoDBSaver {
  getNextVersion(current: number | undefined): number {
    return current === undefined ? 1 : current + 1;
  }
}

function pct(values: number[], p: number) {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil((p / 100) * s.length) - 1)];
}

function channelValue(bytes: number, seed: number) {
  // Messages-like payload: an array of objects with text content.
  const per = 10_000;
  const n = Math.max(1, Math.round(bytes / per));
  return Array.from({ length: n }, (_, i) => ({
    role: i % 2 ? "assistant" : "user",
    content: `${seed}:${i}:`.padEnd(
      per - 40,
      String.fromCharCode(97 + ((seed + i) % 26))
    ),
  }));
}

async function run(
  client: MongoClient,
  dbName: string,
  layout: "old" | "old-nowarn" | "layout2",
  totalBytes: number,
  iterations: number
) {
  let written = 0;
  const params = {
    client,
    dbName,
    // "old-nowarn": the >80% size warning (which re-serializes each channel
    // to report the largest ones) disabled, to isolate the layout cost.
    sizeWarningRatio: layout === "old-nowarn" ? 2 : undefined,
    onDocumentSize: (info: { bytes: number }) => {
      written += info.bytes;
    },
  };
  const saver =
    layout === "layout2"
      ? new MongoDBSaver(params)
      : new OldFormatSaver(params);
  const thread_id = `${layout}-${totalBytes}-${uuid6(-1)}`;
  const values: Record<string, unknown> = { status: "running", step: 0 };
  const versions: ChannelVersions = {};
  const bump = (ch: string) => {
    const max = Object.values(versions)
      .sort((a, b) =>
        typeof a === "number" && typeof b === "number"
          ? a - b
          : String(a).localeCompare(String(b))
      )
      .pop();
    versions[ch] = saver.getNextVersion(max as number | undefined);
    return versions[ch];
  };
  let config = { configurable: { thread_id, checkpoint_ns: "" } } as {
    configurable: Record<string, unknown>;
  };
  const put = async (newVersions: ChannelVersions) => {
    const checkpoint: Checkpoint = {
      ...emptyCheckpoint(),
      id: uuid6(-1),
      channel_values: { ...values },
      channel_versions: { ...versions },
    };
    config = (await saver.put(
      config,
      checkpoint,
      { source: "loop", step: 1, parents: {} },
      newVersions
    )) as typeof config;
  };

  // initial state
  const init: ChannelVersions = {};
  for (let c = 0; c < CHANNELS; c += 1) {
    values[`c${c}`] = channelValue(totalBytes / CHANNELS, c);
    init[`c${c}`] = bump(`c${c}`);
  }
  init.status = bump("status");
  init.step = bump("step");
  await put(init);

  const putMs: number[] = [];
  const writesMs: number[] = [];
  const getMs: number[] = [];
  const bytesPerStep: number[] = [];
  const warmup = 5;
  for (let i = 0; i < iterations + warmup; i += 1) {
    const ch = `c${i % CHANNELS}`;
    values[ch] = channelValue(totalBytes / CHANNELS, i + 100);
    values.step = i + 1;
    const newVersions = { [ch]: bump(ch), step: bump("step") };

    written = 0;
    let t = performance.now();
    await put(newVersions);
    const putTime = performance.now() - t;
    const putBytes = written;

    written = 0;
    t = performance.now();
    await saver.putWrites(
      config as never,
      [
        ["c0", [{ role: "assistant", content: "w".repeat(1000) }]],
        ["status", "ok"],
      ],
      `task-${i}`
    );
    const writesTime = performance.now() - t;
    const writeBytes = written;

    t = performance.now();
    const tuple = await new MongoDBSaver({ client, dbName }).getTuple({
      configurable: { thread_id, checkpoint_ns: "" },
    });
    const getTime = performance.now() - t;
    if (tuple?.checkpoint.channel_values.step !== i + 1)
      throw new Error("bad read");

    if (i >= warmup) {
      putMs.push(putTime);
      writesMs.push(writesTime);
      getMs.push(getTime);
      bytesPerStep.push(putBytes + writeBytes);
    }
  }
  return {
    put: [pct(putMs, 50), pct(putMs, 99)],
    putWrites: [pct(writesMs, 50), pct(writesMs, 99)],
    getTuple: [pct(getMs, 50), pct(getMs, 99)],
    bytesPerStep: bytesPerStep.reduce((a, b) => a + b, 0) / bytesPerStep.length,
  };
}

async function main() {
  const url = process.env.MONGODB_URL ?? process.env.MONGODB_URI;
  if (!url) throw new Error("set MONGODB_URL");
  const args = process.argv.slice(2);
  const it = args.indexOf("--iterations");
  const iterations = it >= 0 ? Number(args[it + 1]) : 50;
  const label = process.env.BENCH_LABEL ?? "";
  const dbName = `ckpt_test_js_bench_${randomBytes(4).toString("hex")}`;
  const client = new MongoClient(url);
  await client.connect();
  const hello = await client.db("admin").command({ hello: 1 });
  const build = await client.db("admin").command({ buildInfo: 1 });
  try {
    await new MongoDBSaver({ client, dbName }).setup();
    const f = (n: number) => n.toFixed(1);
    console.log(
      `\n${label} MongoDB ${build.version} (maxWireVersion ${hello.maxWireVersion}), ${iterations} iterations, ${CHANNELS} channels, 1 changing per step\n`
    );
    console.log(
      "| state | layout | put p50 / p99 (ms) | putWrites p50 / p99 (ms) | getTuple p50 / p99 (ms) | bytes written / step |"
    );
    console.log("|---|---|---|---|---|---|");
    for (const total of [1 * MB, 6 * MB, 15 * MB]) {
      const layouts =
        total >= 15 * MB
          ? (["old", "old-nowarn", "layout2"] as const)
          : (["old", "layout2"] as const);
      for (const layout of layouts) {
        const r = await run(client, dbName, layout, total, iterations);
        console.log(
          `| ${total / MB} MB | ${layout} | ${f(r.put[0])} / ${f(r.put[1])} | ${f(r.putWrites[0])} / ${f(r.putWrites[1])} | ${f(r.getTuple[0])} / ${f(r.getTuple[1])} | ${Math.round(r.bytesPerStep).toLocaleString("en-US")} |`
        );
      }
    }
  } finally {
    if (dbName.startsWith("ckpt_test_js_"))
      await client.db(dbName).dropDatabase();
    await client.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
