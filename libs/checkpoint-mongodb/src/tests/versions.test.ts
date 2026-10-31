import { describe, it, expect } from "vitest";
import { nextVersion, versionKind, VERSION_PATTERN } from "../versions.js";

const GOLDEN = /^\d{32}\.[0-9.]{16}$/;

describe("versions", () => {
  it("TC-037: nextVersion(undefined) returns the string format with prefix 1", () => {
    const v = nextVersion(undefined) as string;
    expect(typeof v).toBe("string");
    expect(v).toMatch(GOLDEN);
    expect(v.slice(0, 32)).toBe("1".padStart(32, "0"));
  });

  it("TC-037: nextVersion(string) increments the prefix and sorts above the input", () => {
    const input = `${"5".padStart(32, "0")}.0.12345678901234`;
    for (let i = 0; i < 200; i += 1) {
      const v = nextVersion(input) as string;
      expect(v).toMatch(GOLDEN);
      expect(v.slice(0, 32)).toBe("6".padStart(32, "0"));
      expect(v > input).toBe(true);
    }
  });

  it("TC-037: accepts the Python InMemorySaver format (longer suffix) as input", () => {
    const v = nextVersion(`${"9".padStart(32, "0")}.0.4340483318289568`);
    expect(v).toMatch(GOLDEN);
    expect((v as string).slice(0, 32)).toBe("10".padStart(32, "0"));
  });

  it("TC-037: the random suffix makes sibling versions distinct (forks)", () => {
    const input = nextVersion(undefined);
    const seen = new Set(Array.from({ length: 1000 }, () => nextVersion(input)));
    expect(seen.size).toBe(1000);
  });

  it("TC-038: nextVersion(number) returns number + 1", () => {
    expect(nextVersion(5)).toBe(6);
    expect(nextVersion(0)).toBe(1);
  });

  it("golden: the exported pattern is the cross-language format", () => {
    expect(VERSION_PATTERN.source).toBe(GOLDEN.source);
  });

  it("versionKind classifies channel_versions", () => {
    expect(versionKind({})).toBe("empty");
    expect(versionKind({ a: 1, b: 2 })).toBe("number");
    expect(versionKind({ a: 1, b: nextVersion(undefined) })).toBe("string");
  });
});
