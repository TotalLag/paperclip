import { chmod, mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { afterEach, expect, test } from "vitest";

import {
  execute,
  parseHermesSessionUsage,
  readHermesSessionUsage,
  resolveHermesProfile,
  resolveHermesUsageDatabasePath,
} from "./execute.js";

const tempDirectories: string[] = [];

async function makeTempDirectory() {
  const directory = await mkdtemp(path.join(os.tmpdir(), "hermes-usage-test-"));
  tempDirectories.push(directory);
  return directory;
}

async function writeSessionDatabase(
  hermesHome: string,
  profile: string,
  sessionId: string,
  inputTokens: string,
  outputTokens: string,
  cachedInputTokens: string,
) {
  const profileDirectory = path.join(hermesHome, "profiles", profile);
  await mkdir(profileDirectory, { recursive: true });
  const databasePath = path.join(profileDirectory, "state.db");
  const result = spawnSync("python3", [
    "-c",
    [
      "import sqlite3, sys",
      "connection = sqlite3.connect(sys.argv[1])",
      "connection.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, estimated_cost_usd REAL, actual_cost_usd REAL)')",
      "connection.execute('INSERT INTO sessions VALUES (?, ?, ?, ?, 3.50, 2.50)', tuple(sys.argv[2:6]))",
      "connection.commit()",
      "connection.close()",
    ].join("\n"),
    databasePath,
    sessionId,
    inputTokens,
    outputTokens,
    cachedInputTokens,
  ], { encoding: "utf8" });
  expect(result.status).toBe(0);
  return databasePath;
}

async function writeFakeHermes(directory: string, sessionId: string) {
  const command = path.join(directory, "fake-hermes");
  await writeFile(
    command,
    `#!/bin/sh\nprintf 'tokens: 1 input 2 output\\ncost: $1.25\\nsession_id: ${sessionId}\\n'\n`,
    "utf8",
  );
  await chmod(command, 0o755);
  return command;
}

function makeContext(config: Record<string, unknown>, logs: string[]) {
  return {
    runId: "run-test",
    agent: {
      id: "agent-test",
      companyId: "company-test",
      name: "Hermes test",
      adapterType: "hermes_local",
      adapterConfig: config,
    },
    runtime: { sessionId: null, sessionParams: null, sessionDisplayId: null, taskKey: null },
    config,
    context: {},
    onLog: async (_stream: "stdout" | "stderr", chunk: string) => { logs.push(chunk); },
  };
}

afterEach(async () => {
  await Promise.all(tempDirectories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

test("reads cumulative Hermes usage from the selected profile database", async () => {
  const directory = await makeTempDirectory();
  const hermesHome = path.join(directory, "hermes-home");
  const sessionId = "session-123";
  await writeSessionDatabase(hermesHome, "work", sessionId, "101", "202", "303");
  const command = await writeFakeHermes(directory, sessionId);
  const logs: string[] = [];

  const result = await execute(makeContext({
    hermesCommand: command,
    cwd: directory,
    profile: "work",
    provider: "anthropic",
    quiet: true,
    env: { HERMES_HOME: hermesHome },
  }, logs));

  expect(result.usage).toEqual({ inputTokens: 101, outputTokens: 202, cachedInputTokens: 303 });
  expect(result.usageBasis).toBe("session_cumulative");
  expect(result.costUsd).toBe(1.25);
  expect(result.resultJson).toMatchObject({
    usage: { inputTokens: 101, outputTokens: 202, cachedInputTokens: 303 },
    usageBasis: "session_cumulative",
    usage_basis: "session_cumulative",
  });
  expect(logs.join("")).not.toContain(sessionId);
});

test("rejects invalid usage rows and profiles without consulting a real Hermes home", async () => {
  expect(parseHermesSessionUsage('{"inputTokens":0,"outputTokens":0,"cachedInputTokens":0}')).toBeNull();
  expect(parseHermesSessionUsage('{"inputTokens":-1,"outputTokens":2,"cachedInputTokens":3}')).toBeNull();
  expect(parseHermesSessionUsage('{"inputTokens":9007199254740992,"outputTokens":2,"cachedInputTokens":3}')).toBeNull();
  expect(parseHermesSessionUsage('{"inputTokens":"1","outputTokens":2,"cachedInputTokens":3}')).toBeNull();
  expect(resolveHermesProfile({ profile: "../escape" })).toEqual({ profile: null, invalid: true });
  expect(resolveHermesProfile({}, ["--profile", "../escape"])).toEqual({ profile: null, invalid: true });
  expect(resolveHermesProfile({ profile: "configured" }, ["--profile", "other"])).toEqual({
    profile: null,
    invalid: true,
  });
  expect(resolveHermesProfile({}, ["--profile", "extra"])).toEqual({ profile: "extra", invalid: false });
  expect(resolveHermesUsageDatabasePath("extra", { HERMES_HOME: "relative" })).toBeNull();
});

test("uses argv-only Python invocation and leaves a missing session successful", async () => {
  const calls: unknown[][] = [];
  const fakeSpawn = ((...args: unknown[]) => {
    calls.push(args);
    return { status: 0, signal: null, error: undefined, stdout: '{"inputTokens":1,"outputTokens":2,"cachedInputTokens":3}' };
  }) as unknown as typeof spawnSync;
  const usage = readHermesSessionUsage("/safe/home/state.db", "session-123", fakeSpawn);
  expect(usage).toEqual({ inputTokens: 1, outputTokens: 2, cachedInputTokens: 3 });
  expect(calls[0]?.[0]).toBe("python3");
  expect(calls[0]?.[1]).toEqual(expect.arrayContaining(["/safe/home/state.db", "session-123"]));
  expect(calls[0]?.[2]).toMatchObject({ shell: false });

  const directory = await makeTempDirectory();
  const hermesHome = path.join(directory, "empty-hermes-home");
  const sessionId = "session-missing";
  const command = await writeFakeHermes(directory, sessionId);
  const logs: string[] = [];
  const result = await execute(makeContext({
    hermesCommand: command,
    cwd: directory,
    provider: "anthropic",
    quiet: true,
    env: { HERMES_HOME: hermesHome },
  }, logs));

  expect(result.exitCode).toBe(0);
  expect(result.errorMessage).toBeUndefined();
  expect(result.usage).toEqual({ inputTokens: 1, outputTokens: 2 });
  expect(result.usageBasis).toBe("per_run");
  expect(logs.join("")).not.toContain(hermesHome);
  expect(logs.join("")).not.toContain(sessionId);
});

test("rejects malformed saved session and profile before starting Hermes", async () => {
  const directory = await makeTempDirectory();
  const marker = path.join(directory, "started");
  const command = path.join(directory, "fake-hermes");
  await writeFile(command, `#!/bin/sh\ntouch '${marker}'\n`, "utf8");
  await chmod(command, 0o755);
  const logs: string[] = [];
  const config = { hermesCommand: command, cwd: directory, provider: "anthropic", profile: "../escape" };

  const profileResult = await execute(makeContext(config, logs));
  expect(profileResult.errorMessage).toBe("Invalid Hermes profile configuration.");
  expect(spawnSync("test", ["-e", marker]).status).not.toBe(0);

  const sessionResult = await execute({
    ...makeContext({ ...config, profile: "safe" }, logs),
    runtime: { sessionId: null, sessionParams: { sessionId: "../escape" }, sessionDisplayId: null, taskKey: null },
  });
  expect(sessionResult.errorMessage).toBe("Invalid Hermes session identifier.");
  expect(spawnSync("test", ["-e", marker]).status).not.toBe(0);
});
