import { spawnSync } from "node:child_process";
import { chmod, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { expect, test } from "vitest";

import type { AdapterExecutionContext } from "@paperclipai/adapter-utils";

import {
  execute,
  isHermesUsageDatabasePathSafe,
  normalizeHermesProfileArgs,
  resolveHermesProfile,
  resolveHermesUsageDatabasePath,
} from "./execute.js";

function executionContext(config: Record<string, unknown>, logs: string[]): AdapterExecutionContext {
  return {
    runId: "run-test",
    agent: {
      id: "agent-test",
      companyId: "company-test",
      name: "Hermes test agent",
      adapterType: "hermes_local",
      adapterConfig: config,
    },
    runtime: {
      sessionId: null,
      sessionParams: null,
      sessionDisplayId: null,
      taskKey: null,
    },
    config,
    context: {},
    onLog: async (_stream, chunk) => {
      logs.push(chunk);
    },
  };
}

async function writeExecutable(filePath: string, body: string): Promise<void> {
  await writeFile(filePath, body, "utf8");
  await chmod(filePath, 0o755);
}

test("normalizes the live LazyP profile shape to one Hermes selector", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "hermes-profile-"));
  const hermesPath = path.join(tempDir, "fake-hermes");
  const argsPath = path.join(tempDir, "args");
  const logs: string[] = [];

  try {
    await writeExecutable(
      hermesPath,
      "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_ARGS_FILE\"\nprintf 'ok\\n'\n",
    );
    const result = await execute(executionContext({
      hermesCommand: hermesPath,
      profile: "lazyp-operator-1",
      extraArgs: ["--profile", "lazyp-operator-1", "--some-flag"],
      env: { FAKE_ARGS_FILE: argsPath, HERMES_HOME: path.join(tempDir, "home") },
    }, logs));
    const args = (await readFile(argsPath, "utf8")).trim().split("\n");
    const selectors = args.filter((arg) => arg === "--profile" || arg.startsWith("--profile="));

    expect(result.exitCode).toBe(0);
    expect(selectors).toEqual(["--profile"]);
    expect(args[args.indexOf("--profile") + 1]).toBe("lazyp-operator-1");
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("rejects conflicting profile selectors before spawning Hermes", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "hermes-profile-conflict-"));
  const hermesPath = path.join(tempDir, "fake-hermes");
  const spawnedPath = path.join(tempDir, "spawned");
  const logs: string[] = [];

  try {
    await writeExecutable(hermesPath, "#!/bin/sh\n: > \"$FAKE_SPAWNED\"\n");
    const result = await execute(executionContext({
      hermesCommand: hermesPath,
      profile: "lazyp-operator-1",
      extraArgs: ["--profile=other-operator"],
      env: { FAKE_SPAWNED: spawnedPath },
    }, logs));

    await expect(readFile(spawnedPath, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
    expect(result.errorMessage).toBe("Invalid Hermes profile configuration.");
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("deduplicates matching selectors and accepts equals-form selectors", () => {
  expect(normalizeHermesProfileArgs({}, [
    "--profile=operator-1",
    "--profile",
    "operator-1",
    "--profile=operator-1",
    "--verbose",
  ])).toEqual({
    profile: "operator-1",
    invalid: false,
    extraArgs: ["--verbose"],
  });
  expect(resolveHermesProfile({ profile: "operator-1" }, ["--profile=operator-1"])).toEqual({
    profile: "operator-1",
    invalid: false,
  });
});

test("does not query usage or retain malformed child-output session IDs", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "hermes-malformed-session-"));
  const hermesPath = path.join(tempDir, "fake-hermes");
  const pythonPath = path.join(tempDir, "python3");
  const pythonMarkerPath = path.join(tempDir, "python-called");
  const malformedSessionId = "../../outside";
  const logs: string[] = [];

  try {
    await writeExecutable(
      hermesPath,
      `#!/bin/sh\nprintf 'answer\\nsession_id: ${malformedSessionId}\\n'\n`,
    );
    await writeExecutable(pythonPath, "#!/bin/sh\n: > \"$FAKE_PYTHON_MARKER\"\n");
    const result = await execute(executionContext({
      hermesCommand: hermesPath,
      quiet: true,
      env: {
        PATH: tempDir,
        HERMES_HOME: path.join(tempDir, "home"),
        FAKE_PYTHON_MARKER: pythonMarkerPath,
      },
    }, logs));

    await expect(readFile(pythonMarkerPath, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
    expect(result.exitCode).toBe(0);
    expect(result.resultJson?.session_id).toBeNull();
    expect(JSON.stringify(result)).not.toContain(malformedSessionId);
    expect(logs.join("")).not.toContain(malformedSessionId);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("keeps cumulative database usage authoritative while retaining per-run stdout cost", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "hermes-db-usage-"));
  const hermesPath = path.join(tempDir, "fake-hermes");
  const home = path.join(tempDir, "home");
  const databasePath = path.join(home, "profiles", "operator-1", "state.db");
  const logs: string[] = [];

  try {
    await mkdir(path.dirname(databasePath), { recursive: true });
    const setup = spawnSync("python3", [
      "-c",
      [
        "import sqlite3, sys",
        "connection = sqlite3.connect(sys.argv[1])",
        "connection.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER)')",
        "connection.execute('INSERT INTO sessions VALUES (?, ?, ?, ?)', ('session-1', 100, 200, 3))",
        "connection.commit()",
        "connection.close()",
      ].join("\n"),
      databasePath,
    ], { encoding: "utf8" });
    expect(setup.status).toBe(0);
    await writeExecutable(
      hermesPath,
      "#!/bin/sh\nprintf 'tokens: 12 input 34 output\\ncost: $0.56\\nsession_id: session-1\\n'\n",
    );
    const result = await execute(executionContext({
      hermesCommand: hermesPath,
      profile: "operator-1",
      env: { HERMES_HOME: home },
    }, logs));

    expect(result.usage).toEqual({ inputTokens: 100, outputTokens: 200, cachedInputTokens: 3 });
    expect(result.usageBasis).toBe("session_cumulative");
    expect(result.costUsd).toBe(0.56);
    expect(result.resultJson).toMatchObject({
      usageBasis: "session_cumulative",
      usage_basis: "session_cumulative",
      cost_usd: 0.56,
    });
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("marks stdout-only usage as per-run without changing stdout cost", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "hermes-stdout-usage-"));
  const hermesPath = path.join(tempDir, "fake-hermes");
  const logs: string[] = [];

  try {
    await writeExecutable(hermesPath, "#!/bin/sh\nprintf 'tokens: 12 input 34 output\\ncost: $0.56\\n'\n");
    const result = await execute(executionContext({ hermesCommand: hermesPath }, logs));

    expect(result.usage).toEqual({ inputTokens: 12, outputTokens: 34 });
    expect(result.usageBasis).toBe("per_run");
    expect(result.costUsd).toBe(0.56);
    expect(result.resultJson).toMatchObject({
      usage: { inputTokens: 12, outputTokens: 34 },
      usageBasis: "per_run",
      usage_basis: "per_run",
      cost_usd: 0.56,
    });
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("contains usage reads under HERMES_HOME and rejects profile traversal and symlink escapes", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "hermes-usage-path-"));
  const home = path.join(tempDir, "custom-home");
  const outside = path.join(tempDir, "outside.db");
  const databasePath = path.join(home, "profiles", "operator-1", "state.db");

  try {
    await writeFile(outside, "not a database", "utf8");
    await mkdir(path.dirname(databasePath), { recursive: true });
    await symlink(outside, databasePath);

    expect(resolveHermesProfile({ profile: "../outside" })).toEqual({ profile: null, invalid: true });
    expect(resolveHermesUsageDatabasePath("../outside", { HERMES_HOME: home })).toBeNull();
    expect(resolveHermesUsageDatabasePath("operator-1", {
      HERMES_HOME: `${home}${path.sep}..${path.sep}custom-home`,
    })).toBeNull();
    expect(resolveHermesUsageDatabasePath("operator-1", {
      HERMES_HOME: home,
      usageDatabasePath: outside,
    })).toBe(databasePath);
    await expect(isHermesUsageDatabasePathSafe(databasePath, { HERMES_HOME: home })).resolves.toBe(false);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
});
