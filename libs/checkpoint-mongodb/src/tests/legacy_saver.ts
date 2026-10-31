/* eslint-disable import/no-extraneous-dependencies */
import { pathToFileURL } from "node:url";
import { join } from "node:path";
import type { MongoClient } from "mongodb";
import type { BaseCheckpointSaver } from "@langchain/langgraph-checkpoint";

/**
 * Loads the PUBLISHED `@langchain/langgraph-checkpoint-mongodb@1.4.2` from
 * the directory in `LEGACY_SAVER_DIR` (an `npm install`ed temp dir), so tests
 * can write genuine old-format threads. Returns undefined when unset.
 */
export async function loadLegacySaver(): Promise<
  | (new (params: { client: MongoClient; dbName?: string }) => BaseCheckpointSaver)
  | undefined
> {
  const dir = process.env.LEGACY_SAVER_DIR;
  if (!dir) return undefined;
  const pkg = join(
    dir,
    "node_modules/@langchain/langgraph-checkpoint-mongodb/dist/index.js"
  );
  const mod = await import(/* @vite-ignore */ pathToFileURL(pkg).href);
  return mod.MongoDBSaver;
}
