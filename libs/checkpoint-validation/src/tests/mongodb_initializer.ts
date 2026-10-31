/* eslint-disable import/no-extraneous-dependencies */
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import {
  MongoDBContainer,
  type StartedMongoDBContainer,
} from "@testcontainers/mongodb";

import { MongoClient } from "mongodb";
import type { CheckpointerTestInitializer } from "../types.js";

const dbName = "test_db";

// Both MongoDB write paths must pass the suite (REQ-E-041): MongoDB 8.0+
// uses one client-level bulkWrite, 7.x an ordered collection bulkWrite plus
// the manifest upsert. Select the server with MONGODB_IMAGE (e.g. mongo:7).
// eslint-disable-next-line no-process-env
const image = process.env.MONGODB_IMAGE ?? "mongo:8";

const container = new MongoDBContainer(image);

let startedContainer: StartedMongoDBContainer;
let client: MongoClient | undefined;

/**
 * Stores every checkpoint in the per-channel layout, including the integer
 * channel versions the conformance fixtures use (which the default saver
 * keeps in the single-document format for threads written by older releases).
 */
class Layout2MongoDBSaver extends MongoDBSaver {
  protected useLayout2(): boolean {
    return true;
  }
}

function createInitializer(
  forceLayout2: boolean
): CheckpointerTestInitializer<MongoDBSaver> {
  return {
  checkpointerName: "@langchain/langgraph-checkpoint-mongodb",

  async beforeAll() {
    startedContainer = await container.start();
    const connectionString = `mongodb://127.0.0.1:${startedContainer.getMappedPort(
      27017
    )}/${dbName}?directConnection=true`;
    client = new MongoClient(connectionString);
  },

  beforeAllTimeout: 300_000, // five minutes, to pull docker container

  async createCheckpointer() {
    // ensure fresh database for each test
    const db = await client!.db(dbName);
    await db.dropDatabase();
    await client!.db(dbName);

    const saver = forceLayout2
      ? new Layout2MongoDBSaver({ client: client! })
      : new MongoDBSaver({ client: client! });
    await saver.setup();
    return saver;
  },

  async afterAll() {
    await client?.close();
    await startedContainer.stop();
  },
  };
}

export const initializer = createInitializer(false);

export const layout2Initializer = createInitializer(true);

export default initializer;
