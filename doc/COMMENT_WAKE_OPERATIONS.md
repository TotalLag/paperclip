# Explicit comment resume: deployment and verification

The fork's runtime branch `fix/hermes-usage-accounting` includes PR #3
(`5d0ccd7863cc56fe861f4475f839a02b67955ea0`). Do not confuse this branch
with the fork's default `master` or a published npm package.

## Behavior

Ordinary board or agent comments on blocked or closed issues no longer implicitly
move them to `todo` or wake their assignee. An intentional restart uses structured
`resume: true` (or the supported explicit reopen operation), subject to existing
blocker, cancellation, ownership and pause checks. Both the comment POST and the
issue PATCH-with-comment routes are covered.

This is not a blanket no-wake flag: existing comment behavior for active work,
mentions, approval transitions and scheduled retries is unchanged. Avoid mentions
in information-only receipts. Do not claim all generic comments are inert.

## Verification

Run the focused suites:

```sh
pnpm exec vitest run server/src/__tests__/issue-comment-reopen-routes.test.ts server/src/__tests__/issue-update-comment-wakeup-routes.test.ts --maxWorkers=1 --testTimeout=20000 --hookTimeout=20000
```

On hosts with unusable IPv6 loopback, the Supertest setup's wildcard-to-`[::1]`
connection can stall before the handler executes. Confirm the listener/client
socket evidence first. A temporary test-only Supertest address override to
`127.0.0.1` and a bounded 20-second test timeout allow the unchanged route suites
to execute; the cold route import can exceed the default five seconds. Keep
this workaround out of production code and report it explicitly. Two suites
cover 80 tests at the reviewed revision. Mock external-object sync warnings are
not evidence of a real database connection or successful integration.

The fork's Dependency Review action currently reports that the GitHub feature
is unsupported. Report that check as failed; do not relabel it green or treat it
as a route regression. Keep local executable evidence separate.

## Narrow installed-artifact rollout

Prefer a package built from the reviewed runtime branch. If applying the exact
compiled route delta to an existing installation, retain the pre-change file,
permissions and SHA-256, reviewed source revision, staged file SHA-256 and exact
diff. Change only `@paperclipai/server/dist/routes/issues.js`; no database/schema,
provider, credential, adapter or scheduler mutation is required.

Require zero live runs across **all companies** before replacing the artifact
and restarting the existing managed Paperclip service. Recheck immediately before
restart; do not interrupt agents or launch an unmanaged second server. Verify
service health, authenticated company/agent/project/issue reads and UI response.
Then record an information-only comment without mentions against a known blocked
issue, read its status and comment back, and verify no issue run was created.
Positive resume/deduplication proof belongs in the focused route suites unless
an isolated live test has separately bounded execution and cleanup.

Rollback: after the same drain gate, restore the retained original file only if
its hash matches the manifest, restart the existing service, and repeat health
and authenticated readback. No database rollback is needed for this route-only
change. A package reinstall/upgrade may overwrite a local artifact patch: inspect
the route and repeat the comment canary after every replacement until the shipped
package includes the reviewed fix.
