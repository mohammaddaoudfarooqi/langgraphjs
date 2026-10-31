import { specTest } from "../spec/index.js";
import { deltaChannelHistoryTests } from "../spec/delta_channel_history.js";
import { layout2Initializer } from "./mongodb_initializer.js";
import { isSkippedCIEnvironment } from "./utils.js";

// The MongoDB saver with every checkpoint in the per-channel layout (layout 2).
if (isSkippedCIEnvironment()) {
  it.skip(`${layout2Initializer.checkpointerName} (layout 2) skipped in CI because no container runtime is available`, () => {});
} else {
  specTest(layout2Initializer);
  deltaChannelHistoryTests(layout2Initializer);
}
