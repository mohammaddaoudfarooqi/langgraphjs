/* eslint-disable import/no-extraneous-dependencies */
import { randomBytes } from "node:crypto";
import { MongoClient, type MongoClientOptions } from "mongodb";

/**
 * Shared helpers for the integration tests added with the per-channel layout.
 *
 * - The server URL comes from `MONGODB_URL` and is used as-is (no forced
 *   credentials), so the same tests run against a local replica set, a local
 *   8.x container, or Atlas.
 * - Every test file gets its own database whose name starts with
 *   `ckpt_test_js_`; the database is dropped in `afterAll`. Nothing else on the
 *   server is ever touched.
 */
export const DB_PREFIX = "ckpt_test_js_";

export function mongoUrl(): string {
  const url = process.env.MONGODB_URL;
  if (!url) {
    throw new Error("MONGODB_URL must be set for integration tests");
  }
  return url;
}

export function uniqueDbName(label: string): string {
  const name = `${DB_PREFIX}${label}_${randomBytes(4).toString("hex")}`;
  assertTestDbName(name);
  return name;
}

export function assertTestDbName(name: string): void {
  if (!name.startsWith(DB_PREFIX)) {
    throw new Error(`refusing to use non-test database ${name}`);
  }
}

export function newClient(options: MongoClientOptions = {}): MongoClient {
  return new MongoClient(mongoUrl(), options);
}

export async function dropTestDb(client: MongoClient, name: string) {
  assertTestDbName(name);
  await client.db(name).dropDatabase();
}

export async function serverInfo(client: MongoClient) {
  const hello = await client.db("admin").command({ hello: 1 });
  return {
    maxWireVersion: hello.maxWireVersion as number,
    isReplicaSet: hello.setName !== undefined || hello.msg === "isdbgrid",
  };
}
