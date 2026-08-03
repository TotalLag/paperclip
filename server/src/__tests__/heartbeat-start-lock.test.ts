import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { withAgentStartLock } from "../services/agent-start-lock.ts";

describe("heartbeat agent start lock", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not let a stale start lock freeze later queued-run starts", async () => {
    vi.useFakeTimers();

    const agentId = randomUUID();
    const firstStart = vi.fn(() => new Promise<void>(() => undefined));
    const secondStart = vi.fn(async () => "started");

    void withAgentStartLock(agentId, firstStart);
    await Promise.resolve();
    expect(firstStart).toHaveBeenCalledTimes(1);

    const secondStartResult = withAgentStartLock(agentId, secondStart);
    await Promise.resolve();
    expect(secondStart).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(30_000);

    await expect(secondStartResult).resolves.toBe("started");
    expect(secondStart).toHaveBeenCalledTimes(1);
  });

  it("ages chained waiters from the oldest lock instead of restarting the stale timeout", async () => {
    vi.useFakeTimers();

    const agentId = randomUUID();
    const firstStart = vi.fn(() => new Promise<void>(() => undefined));
    const secondStart = vi.fn(async () => "second");
    const recoveryStart = vi.fn(async () => "recovered");

    void withAgentStartLock(agentId, firstStart);
    await Promise.resolve();
    const secondStartResult = withAgentStartLock(agentId, secondStart);

    await vi.advanceTimersByTimeAsync(29_000);
    const recoveryStartResult = withAgentStartLock(agentId, recoveryStart);
    await Promise.resolve();
    expect(recoveryStart).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1_000);

    await expect(secondStartResult).resolves.toBe("second");
    await expect(recoveryStartResult).resolves.toBe("recovered");
    expect(recoveryStart).toHaveBeenCalledTimes(1);
  });
});
