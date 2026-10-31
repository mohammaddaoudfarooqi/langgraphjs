import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { CommandStartedEvent, MongoClient } from "mongodb";
import {
  BaseCheckpointSaver,
  DeltaSnapshot,
  emptyCheckpoint,
  uuid6,
} from "@langchain/langgraph-checkpoint";
import type {
  ChannelVersions,
  Checkpoint,
  DeltaChannelHistory,
} from "@langchain/langgraph-checkpoint";
import type { RunnableConfig } from "@langchain/core/runnables";
import { MongoDBSaver, CheckpointIntegrityError } from "../index.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

let client: MongoClient;
const dbName = uniqueDbName("history");
const events: CommandStartedEvent[] = [];

beforeAll(async () => {
  client = newClient({ monitorCommands: true });
  client.on("commandStarted", (e) => events.push(e));
  await client.connect();
  await new MongoDBSaver({ client, dbName }).setup();
});

afterAll(async () => {
  await dropTestDb(client, dbName);
  await client.close();
});

/** Old-format threads: integer versions (the released package's behaviour). */
class IntegerVersionSaver extends MongoDBSaver {
  getNextVersion(current: number | undefined): number {
    return current === undefined ? 1 : current + 1;
  }
}

const base = (saver: MongoDBSaver, config: RunnableConfig, channels: string[]) =>
  BaseCheckpointSaver.prototype.getDeltaChannelHistory.call(saver, {
    config,
    channels,
  }) as Promise<Record<string, DeltaChannelHistory>>;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

interface Node {
  config: RunnableConfig;
  versions: ChannelVersions;
}

async function putNode(
  saver: MongoDBSaver,
  parent: Node | { config: RunnableConfig; versions: ChannelVersions },
  values: Record<string, unknown>
): Promise<Node> {
  const max = Object.values(parent.versions).sort((a, b) =>
    typeof a === "number" && typeof b === "number"
      ? a - b
      : String(a).localeCompare(String(b))
  ).pop();
  const newVersions: ChannelVersions = {};
  const versions = { ...parent.versions };
  for (const ch of Object.keys(values)) {
    newVersions[ch] = saver.getNextVersion(max as number | undefined);
    versions[ch] = newVersions[ch];
  }
  const checkpoint: Checkpoint = {
    ...emptyCheckpoint(),
    id: uuid6(-1),
    channel_values: values,
    channel_versions: versions,
  };
  const config = await saver.put(
    parent.config,
    checkpoint,
    { source: "loop", step: 1, parents: {} },
    newVersions
  );
  return { config, versions };
}

describe("getDeltaChannelHistory override", () => {
  it("TC-080: equals the base implementation on 10 generated histories (forks, subgraphs, overwrites, snapshots)", async () => {
    const channels = ["messages", "other", "absent"];
    let compared = 0;
    for (let h = 0; h < 10; h += 1) {
      const r = rng(1000 + h);
      const saver =
        h % 3 === 0
          ? new IntegerVersionSaver({ client, dbName })
          : new MongoDBSaver({ client, dbName });
      const checkpoint_ns = h % 2 === 0 ? "" : `sub:${h}`;
      const root = {
        config: { configurable: { thread_id: `h${h}-${uuid6(-1)}`, checkpoint_ns } },
        versions: {},
      };
      const nodes: Node[] = [];
      for (let step = 0; step < 16; step += 1) {
        const parent =
          nodes.length === 0
            ? root
            : r() < 0.3
              ? nodes[Math.floor(r() * nodes.length)] // fork
              : nodes[nodes.length - 1];
        const values: Record<string, unknown> = {};
        const k = r();
        if (step === 0 || k < 0.15) values.messages = new DeltaSnapshot([`snap${step}`]);
        else if (k < 0.25) values.messages = [`overwrite${step}`, { n: step }];
        const o = r();
        if (o < 0.2) values.other = `inline${step}`;
        else if (o < 0.35) values.other = { obj: step };
        else if (o < 0.4) values.other = null;
        const node = await putNode(saver, parent, values);
        nodes.push(node);
        const tasks = ["t-c", "t-a", "t-b"].slice(0, Math.floor(r() * 4));
        for (const task of tasks) {
          const writes: [string, unknown][] = [];
          if (r() < 0.8) writes.push(["messages", [`${task}@${step}`]]);
          if (r() < 0.5) writes.push(["other", `${task}-other@${step}`]);
          if (r() < 0.3) writes.push(["unrelated", 1]);
          if (r() < 0.5) writes.push(["messages", [`${task}#2@${step}`]]);
          if (writes.length > 0) await saver.putWrites(node.config, writes, task);
        }
      }
      // Every node is a target; compared in parallel chunks (the base walk
      // issues one getTuple per ancestor, which is slow over a WAN).
      for (let i = 0; i < nodes.length; i += 8) {
        await Promise.all(
          nodes.slice(i, i + 8).map(async (node) => {
            const [expected, actual] = await Promise.all([
              base(saver, node.config, channels),
              saver.getDeltaChannelHistory({ config: node.config, channels }),
            ]);
            expect(actual).toEqual(expected);
            compared += 1;
          })
        );
      }
      // latest checkpoint (no checkpoint_id in the config)
      const latest = { configurable: { ...root.config.configurable } };
      expect(await saver.getDeltaChannelHistory({ config: latest, channels })).toEqual(
        await base(saver, latest, channels)
      );
    }
    expect(compared).toBe(160);
    // unknown thread / empty channels
    const saver = new MongoDBSaver({ client, dbName });
    const unknown = { configurable: { thread_id: "nope", checkpoint_ns: "" } };
    expect(await saver.getDeltaChannelHistory({ config: unknown, channels })).toEqual(
      await base(saver, unknown, channels)
    );
    expect(await saver.getDeltaChannelHistory({ config: unknown, channels: [] })).toEqual({});
  });

  it("TC-081: a 1,000-step history uses at most 22 queries and loads no payloads except the seed", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const root = {
      config: { configurable: { thread_id: uuid6(-1), checkpoint_ns: "" } },
      versions: {},
    };
    let node = await putNode(saver, root, { messages: new DeltaSnapshot(["seed"]) });
    const bulk: Record<string, unknown>[] = [];
    for (let i = 1; i < 1000; i += 1) {
      await saver.putWrites(node.config, [["messages", [`m${i}`]]], "task");
      node = await putNode(saver, node, i % 7 === 0 ? { other: { i } } : {});
    }
    void bulk;
    events.length = 0;
    const hist = await saver.getDeltaChannelHistory({
      config: node.config,
      channels: ["messages"],
    });
    const finds = events.filter((e) => e.commandName === "find");
    const getMores = events.filter((e) => e.commandName === "getMore");
    process.stderr.write(
      `TC-081: ${finds.length} find + ${getMores.length} getMore commands\n`
    );
    expect(finds.length).toBeLessThanOrEqual(10 + 10 + 2);
    expect(getMores.length).toBe(0);
    // Ancestor scans never fetch the checkpoint payload.
    const manifestFinds = finds.filter((e) => e.command.find === "checkpoints");
    for (const f of manifestFinds) {
      expect(f.command.projection).toBeDefined();
      expect(f.command.projection.checkpoint).toBeUndefined();
    }
    expect((hist.messages.seed as DeltaSnapshot).value).toEqual(["seed"]);
    expect(hist.messages.writes).toHaveLength(999);
    expect(hist.messages.writes[0][2]).toEqual(["m1"]);
    expect(hist.messages.writes[998][2]).toEqual(["m999"]);
  }, 900_000);

  it("TC-083: a missing ancestor raises CheckpointIntegrityError (the base silently truncates)", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const root = {
      config: { configurable: { thread_id: uuid6(-1), checkpoint_ns: "" } },
      versions: {},
    };
    let node = await putNode(saver, root, { messages: new DeltaSnapshot([0]) });
    const ids: string[] = [];
    for (let i = 1; i < 6; i += 1) {
      await saver.putWrites(node.config, [["messages", [i]]], "task");
      node = await putNode(saver, node, {});
      ids.push(node.config.configurable!.checkpoint_id);
    }
    await client
      .db(dbName)
      .collection("checkpoints")
      .deleteOne({ checkpoint_id: ids[1] });
    const truncated = await base(saver, node.config, ["messages"]);
    expect(truncated.messages.seed).toBeUndefined();
    const err = await saver
      .getDeltaChannelHistory({ config: node.config, channels: ["messages"] })
      .catch((e) => e);
    expect(err).toBeInstanceOf(CheckpointIntegrityError);
    expect(err.checkpointId).toBe(ids[2]);
    expect(err.missing).toEqual([{ checkpoint_id: ids[1] }]);
  });

  it("TC-082: history writes are returned in (task_path, task_id, idx) order", async () => {
    const saver = new MongoDBSaver({ client, dbName });
    const thread_id = uuid6(-1);
    const root = { config: { configurable: { thread_id, checkpoint_ns: "" } }, versions: {} };
    const c0 = await putNode(saver, root, { messages: new DeltaSnapshot([]) });
    const c1 = await putNode(saver, c0, {});
    // Inserted directly, in reverse order, with task_path (as written by
    // savers that store it).
    const { BSON } = await import("mongodb");
    const value = (v: unknown) => new BSON.Binary(new TextEncoder().encode(JSON.stringify(v)));
    const rows = [
      { task_path: "~b", task_id: "a", idx: 1, v: "b-a-1" },
      { task_path: "~b", task_id: "a", idx: 0, v: "b-a-0" },
      { task_path: "~a", task_id: "z", idx: 0, v: "a-z-0" },
      { task_path: "~a", task_id: "y", idx: 0, v: "a-y-0" },
    ];
    await client
      .db(dbName)
      .collection("checkpoint_writes")
      .insertMany(
        rows.map((r) => ({
          thread_id,
          checkpoint_ns: "",
          checkpoint_id: c0.config.configurable!.checkpoint_id,
          task_id: r.task_id,
          task_path: r.task_path,
          idx: r.idx,
          channel: "messages",
          type: "json",
          value: value([r.v]),
        }))
      );
    const hist = await saver.getDeltaChannelHistory({
      config: c1.config,
      channels: ["messages"],
    });
    expect(hist.messages.writes.map((w) => (w[2] as string[])[0])).toEqual([
      "a-y-0",
      "a-z-0",
      "b-a-0",
      "b-a-1",
    ]);
  });
});
