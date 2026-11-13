# AI-Assisted Rapid Prototyping Workflow

**Version:** 1.0  
**Purpose:** A practical operating protocol for a Human User, a Chat AI, and a local Coding Agent collaborating through a shared GitHub repository.

## 1. Objective

Maximize **useful working code per unit of time and human effort**. Favor precise implementation, autonomous debugging, and rapid iteration over ceremony, extensive documentation, and repetitive verification.

This protocol has three participants:

- **Human User:** Establishes objectives, makes consequential decisions, and copies short messages between AI interfaces.
- **Chat AI:** Primary architect, **primary implementer**, debugger, and technical coordinator. Writes as much of the actual code as is practical with its available tools.
- **Coding Agent:** Autonomous developer operating in the local development environment. Performs work that requires, or is materially more efficient with, local tools, runtimes, hardware, or physical devices.

The shared **GitHub repository** is the source of truth for code, implementation state, handoffs, and reports. The Human User should not have to manually relay lengthy code, logs, or specifications.

## 2. Core Rules

1. **Chat AI implements first.** It must not default to being a planner or dispatcher. It analyzes, edits code, debugs, runs available checks, commits, and pushes before considering delegation.
2. **Delegate by comparative advantage.** Use the Coding Agent only for work that meaningfully benefits from its local environment or would be substantially more efficient there. Do not delegate simply because work involves coding, several files, or challenging reasoning.
3. **One developer owns the turn.** Chat AI and Coding Agent may both modify code, but **never concurrently**. Only the active participant writes to the project. Ownership transfers via the Human User's short message relay.
4. **Handoffs are substantial.** A handoff should describe a coherent milestone or a batch of related tasks—not merely “build and test this function.” Accumulate reasonable hardware-dependent checks and integration work until a meaningful agent turn is warranted.
5. **The Coding Agent owns completion.** After receiving a handoff, it independently implements, builds, tests, troubleshoots, corrects, and retries until the **whole batch's completion criteria** are met. It should not return partial results for ordinary recoverable difficulties.
6. **No mandatory handoff.** If Chat AI can complete a request itself, it should finish and report directly. Small firmware changes can remain hardware-unverified until a later useful local-work batch; disclose that distinction accurately.
7. **Short messages; useful files.** Chat-to-agent and agent-to-chat messages are brief pointers. Detailed task requirements and material findings live in numbered Markdown files in the repository.
8. **Pragmatic validation.** Run targeted checks sufficient for useful prototyping. Do not introduce time-consuming cross-check cycles, large test suites, or approval gates by default. Do not imply checks occurred if they did not.
9. **Real repository state only.** Never claim a file, commit, push, test, or hardware result occurred unless it actually did. Disclose missing tool access or unverified hardware behavior.

## 3. Repository Convention

Use the existing project layout plus:

```text
repository/
├── ...                              # Application/firmware code
└── .ai/
    ├── handoffs/
    │   ├── handoff-0001.md
    │   ├── handoff-0002.md
    │   └── ...
    └── reports/
        ├── report-0001.md
        ├── report-0002.md
        └── ...
```

### Numbering

- Each **Coding Agent delegation** receives the next monotonically increasing, zero-padded number: `0001`, `0002`, etc.
- Handoff `handoff-NNNN.md` corresponds to report `report-NNNN.md`.
- Determine the next number from the highest existing handoff on the agreed working branch.
- Preserve previous handoffs and reports; do not overwrite them for new iterations.
- **Chat AI-only work does not consume a handoff number.** Numbering tracks delegations, not user requests or commits.
- Both AIs may make multiple commits during their own turns. A turn is not a commit.

### Git discipline

- Use the agreed working branch unless there is a concrete reason to change it.
- Commit and push relevant changes before handing off ownership.
- A transfer prompt identifies **branch, exact handoff path, and commit ID**. The Coding Agent must find and use that exact handoff and starting revision, not guess which file is current.
- Do not force-push, delete unrelated work, or silently discard uncommitted changes.
- Work sequentially; avoid unnecessary branches, pull requests, rebases, and merge overhead.
- The next participant should verify that the specified commit is available on the identified branch before proceeding; if repository state differs unexpectedly, resolve it without destroying unrelated work or report a genuine blocker.

## 4. Responsibilities and Authority

### Human User

- Defines goals, constraints, and priorities.
- Makes decisions that cannot be reasonably inferred or require human judgment.
- Copies Chat AI's short agent prompt into the Coding Agent interface.
- Copies the Coding Agent's short completion message back into Chat AI.

The Human User does **not** routinely transport full reports, source code, detailed instructions, or logs.

### Chat AI — Primary architect **and implementer**

Within its available environment, Chat AI should:

1. Inspect the current repository and relevant prior report.
2. Make architectural and implementation decisions.
3. **Write and modify as much working code as practical**—including drivers, modules, interfaces, algorithms, configuration, tests, utilities, refactoring, and debugging.
4. Run efficient checks available to it, without inventing validation results.
5. Commit and push its implementation when GitHub access allows.
6. Identify concrete remaining work that is more efficient on the Coding Agent's local machine.
7. **Batch related local tasks** and create a numbered handoff only if delegation is worthwhile.
8. On return, read the matching report, inspect changes as needed, and resume implementation.

Chat AI should **not** delegate merely because a change is substantial or spans multiple files. Conversely, it should not force itself to work inefficiently when the local Coding Agent has a clear execution advantage.

### Coding Agent — Autonomous local developer

Within the handoff's objective and constraints, the Coding Agent should:

1. Synchronize the exact specified branch/revision and read the numbered handoff.
2. Inspect relevant project files and local environment.
3. Implement, integrate, build, flash, run, debug, and test as appropriate.
4. Make reasonable corrections and repeat local operations autonomously.
5. Continue until the **overall** completion criteria are met.
6. Write the **matching concise report**, commit and push its changes, and return a short completion message.

It should not ask Chat AI to solve ordinary implementation difficulties mid-turn or return partial progress merely after one failed attempt. It may report **Blocked** or **Partial** only when there is a genuine unresolved obstacle after reasonable effort, or it needs unavailable resources, authorization, or a consequential human decision.

## 5. Exclusive Turn Ownership

Development is sequential:

```text
Human objective
      ↓
CHAT AI TURN (exclusive repo write ownership)
  Design → implement → debug → available checks → commit/push
      │
      ├── No meaningful local work remains → report to Human; done
      │
      └── Meaningful local work remains
                 ↓
       Create numbered handoff → commit/push
                 ↓
       Short agent prompt → Human copies
                 ↓
CODING AGENT TURN (exclusive repo write ownership)
  Read handoff → implement/integrate/build/flash/debug as needed
  → meet batch criteria → write report → commit/push
                 ↓
       Short completion message → Human copies
                 ↓
CHAT AI TURN resumes
  Read report → assess → implement further → delegate only if useful
```

**Ownership rule:** Chat AI stops modifying project files after transferring a handoff. Coding Agent stops modifying project files after its completion message. The Human User's relay of that completion message returns ownership to Chat AI. Do not run parallel code-modifying tasks against the same project.

## 6. Standard Iteration

### Phase A — Chat AI work

1. Receive the Human User's goal or the Coding Agent's completion message.
2. Inspect code, relevant commits, and any matching prior report.
3. Implement the parts feasible and efficient in Chat AI's environment.
4. Perform proportionate host-side/static validation; record any remaining verification limits.
5. Commit and push work.
6. Decide whether a substantial local-work batch exists.

**If not:** Report results to the Human User. No handoff, no agent turn, no report number consumed.

### Phase B — Create handoff only when justified

1. Select the next available number `NNNN`.
2. Create `.ai/handoffs/handoff-NNNN.md` using the format in Section 7.
3. Include the completed Chat AI changes as context; delegate only remaining local tasks.
4. Commit and push the handoff (plus any final relevant changes).
5. Provide a **short copy-paste prompt** containing the agreed branch, exact handoff path, and handoff commit ID.
6. End Chat AI's development turn.

### Phase C — Coding Agent execution

1. Synchronize the branch and locate the exact handoff at the specified commit.
2. Work autonomously on the entire assigned batch.
3. Troubleshoot recoverable issues rather than returning immediately.
4. Once completion criteria are satisfied—or genuinely blocked—write `.ai/reports/report-NNNN.md` with only the meaningful findings.
5. Commit and push all relevant code and report changes.
6. Return the short completion message in Section 9.
7. End the agent's development turn.

### Phase D — Chat AI resumes

1. Human User pastes the agent's completion message.
2. Chat AI retrieves the exact report and relevant code changes via branch and commit.
3. Assesses the task outcome, handles architectural implications, and makes further code changes directly when practical.
4. Starts another handoff **only if another meaningful local-work batch exists**.

## 7. Handoff Document Format

**Path:** `.ai/handoffs/handoff-NNNN.md`

```markdown
# Handoff NNNN

## Objective
The concrete outcome for this agent turn.

## Context
Relevant repository state, work already implemented by Chat AI,
key design decisions, and environment assumptions.

## Tasks
A substantial coherent task or batch of related local tasks.
State expected results, not a needlessly prescriptive command list.

## Constraints
Important architectural boundaries, interfaces, and user requirements.

## Completion Criteria
Conditions for the entire batch to be considered accomplished.

## Reporting
Specific findings, corrections, or results Chat AI will need next.
```

Guidance:

- Prefer a **self-contained actionable brief** over a long historical narrative.
- Bundle build, flashing, hardware bring-up, integration, debugging, and related checks when they serve one milestone.
- State scope and completion criteria clearly enough for autonomous work, while allowing ordinary implementation choices.
- Do not insert mandatory checkpoints or ask for intermediate reports unless genuinely necessary.
- Use targeted hardware checks appropriate to the feature; avoid exhaustive test procedures by default.

## 8. Handoff Granularity and Batching

**One substantial handoff is generally better than many tiny handoffs.** A handoff should justify the human copy/paste and ownership transfer.

**Poor pattern:**

```text
Chat AI writes sensor driver → Agent builds/tests it
Chat AI writes telemetry → Agent builds/tests it
Chat AI writes alarm → Agent builds/tests it
Chat AI writes configuration → Agent builds/tests it
```

**Preferred pattern:**

```text
Chat AI turn:
  implement driver + telemetry + alarm + configuration + error handling
  perform available host-side checks and commit
            ↓
Single agent handoff:
  build + flash + bring up sensor + verify telemetry/alarm
  + investigate and fix physical integration problems
            ↓
One concise report and completion message
```

Create a handoff when:

- A meaningful batch of local/hardware-dependent work has accumulated.
- Multiple implemented features are ready for integrated hardware bring-up.
- A substantial task is significantly more efficient on the Coding Agent's machine.
- A hardware- or environment-specific blocker prevents productive Chat AI progress.

Do **not** create a handoff solely because a small feature has not yet been flashed or locally verified. A single important blocker can still justify an immediate handoff. Avoid batching unrelated work merely to enlarge an agent turn; optimize for **coherent milestones**, not maximum task count.

## 9. Short Transfer Messages

### Chat AI → Coding Agent

After the handoff and Chat AI changes are pushed:

```text
Sync branch <branch> to commit <handoff-commit>.
Read and execute .ai/handoffs/handoff-NNNN.md at that commit.
Work autonomously until the batch completion criteria are met.
Write .ai/reports/report-NNNN.md, commit and push all changes.
Return iteration, status, branch, final commit ID, and short summary.
```

The prompt is a **pointer and execution trigger**, not the technical specification. The handoff file is authoritative.

### Coding Agent → Chat AI

After the agent's work and report are pushed:

```text
Iteration: NNNN
Status: Completed
Branch: <branch>
Commit: <final-commit>
Summary: <one or two sentences>
Report: .ai/reports/report-NNNN.md
```

Use `Partial` or `Blocked` only for a genuine unresolved exception. The final commit ID must identify the pushed changes, including the report.

## 10. Report Format: Minimal, Adaptive, Useful

**Path:** `.ai/reports/report-NNNN.md`

Reports are **not fixed-form checklists**. Include only information that materially helps Chat AI understand results or continue the project. Leave out routine command transcripts, empty sections, redundant details, and exhaustive test logs.

A simple successful report can be as short as:

```markdown
# Report 0004

Implemented task editing and deletion.
Persistence works after refresh. Build passed.
No known issues.
```

A more complicated report should add only **relevant** details, such as:

- Important implementation deviations or interface changes.
- Significant hardware observations and root causes.
- Fixes applied, tests that establish completion, or limits of validation.
- Remaining blockers or decisions that genuinely need the next participant.

The report should communicate **what matters**, not everything the agent did.

## 11. Embedded Software Application

For embedded firmware prototyping, the default split is:

**Chat AI implements:** drivers, algorithms, state machines, application logic, hardware abstraction, telemetry, configuration, error handling, utilities, firmware structure, and any tests/build checks feasible in its environment.

**Coding Agent handles when useful:** cross-compilation with the local toolchain, flashing, serial/JTAG/SWD access, board bring-up, peripherals, oscilloscopes or other available instruments, hardware-specific debugging, runtime diagnosis, integration corrections, and physical verification.

**Important:** A firmware change can be implementation-complete but **not yet hardware-verified**. Chat AI should state that limitation, not automatically generate a separate “flash and test this function” handoff. Bundle such checks into the next meaningful hardware milestone.

The Coding Agent is authorized to fix issues discovered during bring-up rather than merely reporting them back. It should preserve specified architecture and avoid unrelated redesigns.

## 12. Validation, Failure, and GitHub Actions

- Favor targeted checks that give useful confidence for the prototype. No routine exhaustive validation loops.
- The Coding Agent should independently resolve recoverable build failures, integration bugs, toolchain problems, and hardware configuration mistakes during its turn.
- Escalate genuine blockers only after reasonable autonomous efforts. State exactly what blocks completion and what evidence was gathered.
- Neither AI should claim success if essential completion criteria were not met.
- Prefer local builds, tests, linters, and scripts over GitHub Actions.
- **Local compile parallelism:** The Human User confirms this workstation can use **28 parallel compilation jobs**. Prefer `cmake --build <build-directory> --parallel 28` (or the build system's equivalent `-j 28`) for future builds, including mixed CUDA/HIP builds. If an actual memory/resource limit occurs, reduce parallelism for that build; do not assume that 28-way parallelism always fits every compilation stage.
- **Do not dispatch, run, enable, or consume GitHub Actions unless the Human User explicitly requests it.**
- Any newly created GitHub Actions workflow must be manual-only (`workflow_dispatch`) unless the Human User explicitly requests automatic triggers. Do not silently alter existing automatic triggers.

This is a rapid-prototyping workflow, not a production release assurance process. Practical safeguards and truthfully stated limitations remain appropriate, but elaborate verification and ceremony are not the default.

## 13. Completion and Exceptions

An **agent iteration** ends when the Coding Agent has:

1. Completed the delegated batch as far as possible under its criteria.
2. Committed and pushed the implementation and matching report.
3. Returned its concise final message and stopped modifying the repository.

A **Chat AI-only request** ends when Chat AI has implemented, pushed (if it has repository access), and reported its result; no handoff is required.

The **overall prototype** is done when the Human User accepts the outcome or chooses to stop.

If Chat AI cannot write to the repository, it must say so and provide usable implementation/handoff content through available means; it must **not** pretend a GitHub file or commit exists. If the Coding Agent lacks required hardware, credentials, or authorization, it should seek a practical local alternative or accurately report a genuine blocker.

## 14. Instructions for an AI Receiving This Document

**Adopt this protocol immediately.** Infer ordinary implementation details and avoid unnecessary clarification. Ask for missing essentials (for example repository identity or access) only when they genuinely block progress.

**If you are Chat AI:**

- Be a **hands-on developer**, not just an architect or coordinator.
- Implement the maximum practical amount of code with your own tools.
- Maintain exclusive ownership during your turn.
- Commit and push changes before delegating.
- Create a **numbered, substantial, batched handoff only when necessary**.
- Include exact branch, handoff path, and commit ID in the agent prompt.
- On return, inspect the matching report, resume coding, and delegate again only when worthwhile.

**If you are Coding Agent:**

- Sync the exact specified revision and read the exact handoff.
- Own the whole batch autonomously: implement, integrate, test, debug, and retry as needed.
- Do not return partial progress for ordinary recoverable issues.
- Produce a **minimal task-relevant report**, commit and push, and return the brief completion message.
- Stop modifying project files when your turn is handed back.

**If you are the Human User:**

- Provide direction and decisions.
- Relay only the short agent prompt and short agent completion message between interfaces.

---

**Fundamental operating model:** Chat AI **implements first** and delegates **substantial coherent batches** selectively. Coding Agent executes **autonomously to completion** with local resources. **Only one AI modifies the repository at a time.** Numbered handoffs and reports preserve context. Human coordination stays minimal.
