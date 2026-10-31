/**
 * Converts selected old-format threads to the per-channel layout (layout 2).
 * See src/scripts/migrate_thread_layout.ts for details.
 *
 *   MONGODB_URI=... npx tsx scripts/migrate-thread-layout.ts --db <name> \
 *     --thread <thread_id> [--thread <thread_id> ...] [--keep-backup]
 */
import { migrateThreadLayoutCli } from "../src/scripts/migrate_thread_layout.js";

migrateThreadLayoutCli(process.argv.slice(2)).then(
  (code) => {
    process.exitCode = code;
  },
  (e) => {
    console.error(e);
    process.exitCode = 1;
  }
);
