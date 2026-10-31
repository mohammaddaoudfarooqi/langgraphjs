import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { MongoClient } from "mongodb";
/* eslint-disable import/no-extraneous-dependencies */
import {
  Annotation,
  Command,
  END,
  START,
  StateGraph,
  interrupt,
} from "@langchain/langgraph";
import { uuid6 } from "@langchain/langgraph-checkpoint";
import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";
import { checkUniqueIndexes } from "../scripts/check_unique_indexes.js";
import { migrateThreadLayout } from "../scripts/migrate_thread_layout.js";
import { MongoDBSaver } from "../index.js";
import { VERSION_PATTERN } from "../versions.js";
import { loadLegacySaver } from "./legacy_saver.js";
import { dropTestDb, newClient, uniqueDbName } from "./int_utils.js";

let client: MongoClient;
const dbNames: string[] = [];
function freshDb(label: string) {
  const name = uniqueDbName(label);
  dbNames.push(name);
  return name;
}

beforeAll(async () => {
  client = newClient();
  await client.connect();
});

afterAll(async () => {
  for (const name of dbNames) await dropTestDb(client, name);
  await client.close();
});

describe("scripts/check-unique-indexes", () => {
  it("TC-022: reports duplicates without --apply and changes nothing; --apply on a clean collection rebuilds the index as unique", async () => {
    const dbName = freshDb("script_idx");
    const coll = client.db(dbName).collection("checkpoints");
    await coll.createIndex(
      { thread_id: 1, checkpoint_ns: 1, checkpoint_id: -1 },
      { name: "thread_ns_checkpoint_idx" }
    );
    const row = { thread_id: "t", checkpoint_ns: "", checkpoint_id: "c1" };
    await coll.insertMany([{ ...row, n: 1 }, { ...row, n: 2 }]);
    const indexesBefore = await coll.listIndexes().toArray();

    const report = await checkUniqueIndexes({ client, dbName, apply: false });
    const cpReport = report.find((r) => r.collection === "checkpoints")!;
    expect(cpReport.status).toBe("duplicates");
    expect(cpReport.duplicates).toEqual([
      { key: { thread_id: "t", checkpoint_ns: "", checkpoint_id: "c1" }, count: 2 },
    ]);
    expect(await coll.countDocuments()).toBe(2);
    expect(await coll.listIndexes().toArray()).toEqual(indexesBefore);

    // --apply refuses while duplicates remain.
    const refused = await checkUniqueIndexes({ client, dbName, apply: true });
    expect(refused.find((r) => r.collection === "checkpoints")!.status).toBe(
      "duplicates"
    );
    expect(await coll.listIndexes().toArray()).toEqual(indexesBefore);

    await coll.deleteOne({ n: 2 });
    const applied = await checkUniqueIndexes({ client, dbName, apply: true });
    expect(applied.find((r) => r.collection === "checkpoints")!.status).toBe(
      "rebuilt"
    );
    const after = await coll.listIndexes().toArray();
    expect(after).toContainEqual(
      expect.objectContaining({
        name: "thread_ns_checkpoint_idx",
        key: { thread_id: 1, checkpoint_ns: 1, checkpoint_id: -1 },
        unique: true,
      })
    );

    // Idempotent: a second run is a no-op.
    const again = await checkUniqueIndexes({ client, dbName, apply: true });
    expect(again.find((r) => r.collection === "checkpoints")!.status).toBe(
      "ok"
    );
  });
});

/** Writes old-format threads: the released 1.4.2 package when available. */
class IntegerVersionSaver extends MongoDBSaver {
  getNextVersion(current: number | undefined): number {
    return current === undefined ? 1 : current + 1;
  }
}

const State = Annotation.Root({
  log: Annotation<string[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  data: Annotation<Record<string, unknown>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
  note: Annotation<string | undefined>(),
  answer: Annotation<string | undefined>(),
});

const graph = (saver: BaseCheckpointSaver) =>
  new StateGraph(State)
    .addNode("a", () => ({ log: ["a"], data: { a: "x".repeat(3000) }, note: "n1" }))
    .addNode("b", () => ({ log: ["b"], data: { b: [1, 2, 3] } }))
    .addNode("ask", () => ({ answer: interrupt("q") as string }))
    .addNode("c", (s) => ({ log: [`c:${s.answer}`], data: { c: true } }))
    .addEdge(START, "a")
    .addEdge("a", "b")
    .addEdge("b", "ask")
    .addEdge("ask", "c")
    .addEdge("c", END)
    .compile({ checkpointer: saver });

describe("scripts/migrate-thread-layout", () => {
  it("TC-042: converts an old thread to layout 2; resume after equals resume before; re-run is a no-op; sources stay until verified", async () => {
    const dbName = freshDb("script_migrate");
    const Legacy = await loadLegacySaver();
    const oldSaver = () =>
      Legacy
        ? new Legacy({ client, dbName })
        : new IntegerVersionSaver({ client, dbName });
    const saver = () => new MongoDBSaver({ client, dbName });

    // Two identical old-format threads paused at the interrupt, with a fork.
    const threads = [`m-${uuid6(-1)}`, `ref-${uuid6(-1)}`];
    for (const thread_id of threads) {
      const config = { configurable: { thread_id } };
      await graph(oldSaver()).invoke({ log: ["start"] }, config);
      const s = await graph(oldSaver()).getState(config);
      await graph(oldSaver()).updateState(s.config, { note: "forked" });
    }
    const [migrated, reference] = threads;
    const cfg = (thread_id: string) => ({ configurable: { thread_id } });
    const coll = client.db(dbName).collection("checkpoints");
    const before: unknown[] = [];
    for await (const t of saver().list(cfg(migrated))) before.push(t);

    // Verification failure: nothing is replaced.
    const failed = await migrateThreadLayout({
      client,
      dbName,
      threadIds: [migrated],
      beforeVerify: async ({ blobs }) => {
        await client.db(dbName).collection(blobs).deleteMany({ thread_id: migrated });
      },
    });
    expect(failed[0].status).toBe("verification-failed");
    expect(
      await coll.countDocuments({ thread_id: migrated, layout: 2 })
    ).toBe(0);
    const stillOld: unknown[] = [];
    for await (const t of saver().list(cfg(migrated))) stillOld.push(t);
    expect(stillOld).toEqual(before);

    // Real run.
    const result = await migrateThreadLayout({
      client,
      dbName,
      threadIds: [migrated],
    });
    expect(result[0].status).toBe("migrated");
    expect(result[0].checkpoints).toBe(before.length);
    const docs = await coll.find({ thread_id: migrated }).toArray();
    expect(docs).toHaveLength(before.length);
    expect(docs.every((d) => d.layout === 2)).toBe(true);
    expect(
      await client.db(dbName).collection("checkpoint_blobs").countDocuments({ thread_id: migrated })
    ).toBeGreaterThan(0);

    // Same tuples, except for the version strings (same integer prefixes).
    const after: Awaited<ReturnType<MongoDBSaver["getTuple"]>>[] = [];
    for await (const t of saver().list(cfg(migrated))) after.push(t);
    expect(after).toHaveLength(before.length);
    after.forEach((t, i) => {
      const b = before[i] as typeof t;
      expect(t!.checkpoint.channel_values).toEqual(b!.checkpoint.channel_values);
      expect(t!.metadata).toEqual(b!.metadata);
      expect(t!.pendingWrites).toEqual(b!.pendingWrites);
      expect(t!.parentConfig).toEqual(b!.parentConfig);
      for (const [ch, v] of Object.entries(t!.checkpoint.channel_versions)) {
        expect(v).toMatch(VERSION_PATTERN);
        expect(Number.parseInt(String(v).split(".")[0], 10)).toBe(
          b!.checkpoint.channel_versions[ch]
        );
      }
    });

    // Re-running is a no-op.
    const again = await migrateThreadLayout({ client, dbName, threadIds: [migrated] });
    expect(again[0].status).toBe("already-migrated");

    // Resume after conversion equals resume of the untouched reference thread.
    const outMigrated = await graph(saver()).invoke(new Command({ resume: "ok" }), cfg(migrated));
    const outReference = await graph(saver()).invoke(new Command({ resume: "ok" }), cfg(reference));
    expect(outMigrated).toEqual(outReference);
    expect(outMigrated.log).toEqual(["start", "a", "b", "c:ok"]);
    // The resumed migrated thread keeps writing layout 2 with string versions.
    const latest = await saver().getTuple(cfg(migrated));
    for (const v of Object.values(latest!.checkpoint.channel_versions)) {
      expect(v).toMatch(VERSION_PATTERN);
    }
    expect(await coll.countDocuments({ thread_id: migrated, layout: { $ne: 2 } })).toBe(0);
    // The reference thread was never touched.
    expect(await coll.countDocuments({ thread_id: reference, layout: 2 })).toBe(0);
  });
});
