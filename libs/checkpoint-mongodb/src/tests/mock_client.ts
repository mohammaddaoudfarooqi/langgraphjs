/**
 * A small in-memory stand-in for `MongoClient`, covering the operations the
 * saver uses. Documents are stored as BSON round-trips so reads return the
 * same types as the real driver (e.g. `Binary` for byte arrays).
 * TC-PARITY-001 checks it against a real server.
 */
import { BSON, type MongoClient } from "mongodb";

type Doc = Record<string, unknown>;

const clone = (d: Doc): Doc =>
  BSON.deserialize(BSON.serialize(d, { ignoreUndefined: false }), {
    promoteBuffers: false,
  });

function get(doc: Doc, path: string): unknown {
  return path
    .split(".")
    .reduce<unknown>(
      (v, k) => (v != null && typeof v === "object" ? (v as Doc)[k] : undefined),
      doc
    );
}

function cmp(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  const sa = String(a);
  const sb = String(b);
  return sa < sb ? -1 : sa > sb ? 1 : 0;
}

function eq(a: unknown, b: unknown): boolean {
  if (a === null || a === undefined) return b === null || b === undefined;
  return (
    typeof a === typeof b &&
    JSON.stringify(a) === JSON.stringify(b)
  );
}

function matchValue(value: unknown, cond: unknown): boolean {
  if (cond !== null && typeof cond === "object" && !Array.isArray(cond)) {
    const ops = Object.entries(cond as Doc);
    if (ops.length > 0 && ops.every(([k]) => k.startsWith("$"))) {
      return ops.every(([op, arg]) => {
        switch (op) {
          case "$lt":
            return value !== undefined && cmp(value, arg) < 0;
          case "$lte":
            return value !== undefined && cmp(value, arg) <= 0;
          case "$in":
            return (arg as unknown[]).some((x) => eq(value, x));
          case "$exists":
            return (value !== undefined) === Boolean(arg);
          case "$ne":
            return !eq(value, arg);
          default:
            throw new Error(`mock: unsupported operator ${op}`);
        }
      });
    }
  }
  return eq(value, cond);
}

export function matches(doc: Doc, filter: Doc): boolean {
  return Object.entries(filter).every(([k, cond]) => {
    if (k === "$or") return (cond as Doc[]).some((f) => matches(doc, f));
    return matchValue(get(doc, k), cond);
  });
}

function project(doc: Doc, projection?: Doc): Doc {
  if (!projection) return doc;
  const include = Object.entries(projection).filter(([k, v]) => v && k !== "_id");
  const out: Doc = {};
  if (include.length > 0) {
    for (const [k] of include) if (doc[k] !== undefined) out[k] = doc[k];
    if (projection._id !== 0 && doc._id !== undefined) out._id = doc._id;
    return out;
  }
  for (const [k, v] of Object.entries(doc)) if (projection[k] !== 0) out[k] = v;
  return out;
}

class MockCursor {
  private sortSpec?: [string, number];

  private n?: number;

  constructor(private docs: () => Doc[], private projection?: Doc) {}

  sort(spec: string | Doc, dir?: number) {
    this.sortSpec =
      typeof spec === "string"
        ? [spec, dir ?? 1]
        : (Object.entries(spec)[0] as [string, number]);
    return this;
  }

  limit(n: number) {
    this.n = n;
    return this;
  }

  async toArray(): Promise<Doc[]> {
    let docs = this.docs();
    if (this.sortSpec) {
      const [k, d] = this.sortSpec;
      docs = [...docs].sort((a, b) => d * cmp(get(a, k), get(b, k)));
    }
    if (this.n !== undefined && this.n > 0) docs = docs.slice(0, this.n);
    return docs.map((d) => project(clone(d), this.projection));
  }

  async *[Symbol.asyncIterator]() {
    for (const d of await this.toArray()) yield d;
  }
}

function applyUpdate(doc: Doc, update: Doc, inserting: boolean) {
  for (const [op, fields] of Object.entries(update)) {
    for (const [k, v] of Object.entries(fields as Doc)) {
      switch (op) {
        case "$set":
          doc[k] = v;
          break;
        case "$setOnInsert":
          if (inserting) doc[k] = v;
          break;
        case "$currentDate":
          doc[k] = new Date();
          break;
        case "$max":
          if (doc[k] === undefined || cmp(v, doc[k]) > 0) doc[k] = v;
          break;
        case "$inc":
          doc[k] = ((doc[k] as number) ?? 0) + (v as number);
          break;
        case "$unset":
          delete doc[k];
          break;
        default:
          throw new Error(`mock: unsupported update ${op}`);
      }
    }
  }
}

class MockCollection {
  docs: Doc[] = [];

  private nextId = 0;

  constructor(readonly name: string) {}

  find(filter: Doc = {}, options: { projection?: Doc } = {}) {
    return new MockCursor(
      () => this.docs.filter((d) => matches(d, filter)),
      options.projection
    );
  }

  async findOne(filter: Doc = {}) {
    const [d] = await this.find(filter).limit(1).toArray();
    return d ?? null;
  }

  async updateOne(filter: Doc, update: Doc, options: { upsert?: boolean } = {}) {
    const existing = this.docs.find((d) => matches(d, filter));
    if (existing) {
      applyUpdate(existing, update, false);
      Object.assign(existing, clone(existing));
      return { matchedCount: 1, upsertedCount: 0 };
    }
    if (!options.upsert) return { matchedCount: 0, upsertedCount: 0 };
    const doc: Doc = { _id: `${this.name}-${(this.nextId += 1)}` };
    for (const [k, v] of Object.entries(filter)) {
      if (!k.startsWith("$") && (v === null || typeof v !== "object")) doc[k] = v;
    }
    applyUpdate(doc, update, true);
    this.docs.push(clone(doc));
    return { matchedCount: 0, upsertedCount: 1 };
  }

  async bulkWrite(ops: { updateOne: { filter: Doc; update: Doc; upsert?: boolean } }[]) {
    if (ops.length === 0) throw new Error("Invalid BulkOperation, Batch cannot be empty");
    for (const { updateOne } of ops) {
      await this.updateOne(updateOne.filter, updateOne.update, updateOne);
    }
    return { ok: 1 };
  }

  async insertMany(docs: Doc[]) {
    for (const d of docs) this.docs.push(clone({ _id: `${this.name}-${(this.nextId += 1)}`, ...d }));
  }

  async deleteMany(filter: Doc) {
    const before = this.docs.length;
    this.docs = this.docs.filter((d) => !matches(d, filter));
    return { deletedCount: before - this.docs.length };
  }

  async deleteOne(filter: Doc) {
    const i = this.docs.findIndex((d) => matches(d, filter));
    if (i >= 0) this.docs.splice(i, 1);
    return { deletedCount: i >= 0 ? 1 : 0 };
  }

  async createIndex() {
    return "ok";
  }

  listIndexes() {
    return { toArray: async () => [] };
  }
}

export function createInMemoryClient(maxWireVersion = 21): MongoClient {
  const dbs = new Map<string, Map<string, MockCollection>>();
  const db = (name = "test") => {
    if (!dbs.has(name)) dbs.set(name, new Map());
    const colls = dbs.get(name)!;
    return {
      databaseName: name,
      collection(c: string) {
        if (!colls.has(c)) colls.set(c, new MockCollection(c));
        return colls.get(c)!;
      },
      async command(cmd: Doc) {
        if ("hello" in cmd) return { maxWireVersion, setName: "mock" };
        throw new Error("mock: unsupported command");
      },
    };
  };
  return {
    appendMetadata() {},
    db,
    // No real transactions: the callback runs directly.
    async withSession(fn: (session: unknown) => Promise<unknown>) {
      return fn({ withTransaction: (f: () => Promise<unknown>) => f() });
    },
    async bulkWrite(
      models: { namespace: string; name: string; filter: Doc; update: Doc; upsert?: boolean }[]
    ) {
      for (const m of models) {
        const [d, ...rest] = m.namespace.split(".");
        await db(d).collection(rest.join(".")).updateOne(m.filter, m.update, m);
      }
      return {};
    },
  } as unknown as MongoClient;
}
