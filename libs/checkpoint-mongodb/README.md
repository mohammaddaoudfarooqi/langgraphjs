# @langchain/langgraph-checkpoint-mongodb

Implementation of a [LangGraph.js](https://github.com/langchain-ai/langgraphjs) CheckpointSaver that uses a MongoDB instance.

## Usage

```ts
import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";

const writeConfig = {
  configurable: {
    thread_id: "1",
    checkpoint_ns: ""
  }
};
const readConfig = {
  configurable: {
    thread_id: "1"
  }
};


const client = new MongoClient(process.env.MONGODB_URL);

const checkpointer = new MongoDBSaver({ client });

// NOTE: you need to call .setup() the first time you're using your checkpointer
await checkpointer.setup();

const checkpoint = {
  v: 1,
  ts: "2024-07-31T20:14:19.804150+00:00",
  id: "1ef4f797-8335-6428-8001-8a1503f9b875",
  channel_values: {
    my_key: "meow",
    node: "node"
  },
  channel_versions: {
    __start__: 2,
    my_key: 3,
    "start:node": 3,
    node: 3
  },
  versions_seen: {
    __input__: {},
    __start__: {
      __start__: 1
    },
    node: {
      "start:node": 2
    }
  },
  pending_sends: [],
}

// store checkpoint
await checkpointer.put(writeConfig, checkpoint, {}, {});

// load checkpoint
await checkpointer.get(readConfig);

// list checkpoints
for await (const checkpoint of checkpointer.list(readConfig)) {
  console.log(checkpoint);
}

await client.close();
```

## Query scalar checkpoint values

In the per-channel layout, the default serializer stores versioned scalar values such as booleans, `null`, finite numbers, and short UTF-8 strings directly in `checkpoint_blobs.value`. The `type` field is BSON null for native values. Checkpoints share a channel document while its version stays unchanged. Pending writes can also store native scalars in `checkpoint_writes.value`.

```javascript
db.checkpoint_blobs.find(
  { thread_id: "my-thread", checkpoint_ns: "", channel: "system_prompt", type: { $type: "null" }, value: "You are a helpful assistant." },
  { _id: 0, version: 1 }
);
```

This query can return historical versions. To check current state, match the returned channel and version against the latest checkpoint's `blob_refs`. The example uses the default collection names. Strings use a UTF-8 byte budget of at most 1024, which a smaller `inlineMaxChars` setting lowers. Custom serializers continue to encode values themselves.

Existing checkpoints and serialized pending writes remain readable. The new format does not rewrite historical records. Numeric-only legacy threads must be migrated to use per-channel checkpoint storage. Upgrade all readers before writing native scalars to a shared database. Older savers cannot decode the new scalar records. MongoDB's per-document BSON size limit still applies.

## TTL (Time-To-Live) Support

Automatically expire old checkpoints using MongoDB's TTL indexes:

```ts
import { MongoClient } from "mongodb";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";

const client = new MongoClient(process.env.MONGODB_URL);

// Create checkpointer with 1-hour TTL (in seconds)
const checkpointer = new MongoDBSaver({
  client,
  ttl: 3600,
});

// Create TTL indexes (call during deployment/startup)
await checkpointer.setup();
```

When TTL is enabled:
- An `upserted_at` timestamp is added to each document on every write
- MongoDB automatically deletes documents after the TTL expires
- Each update resets the expiration timer

The `setup()` method creates the required TTL indexes. Call it during application startup or deployment. It is idempotent and handles concurrent calls safely.
