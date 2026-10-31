/**
 * Checks (and with --apply rebuilds) the unique indexes MongoDBSaver needs.
 * See src/scripts/check_unique_indexes.ts for details.
 *
 *   MONGODB_URI=... npx tsx scripts/check-unique-indexes.ts --db <name> [--apply]
 */
import { checkUniqueIndexesCli } from "../src/scripts/check_unique_indexes.js";

checkUniqueIndexesCli(process.argv.slice(2)).then(
  (code) => {
    process.exitCode = code;
  },
  (e) => {
    console.error(e);
    process.exitCode = 1;
  }
);
