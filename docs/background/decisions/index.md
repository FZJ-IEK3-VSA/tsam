# Architecture Decision Records (ADR)

This section collects the **architectural decisions** that shape tsam. Each decision is a short, append-only markdown file capturing *why* a particular design choice was made — the kind of context that's invisible in the code and easily lost from `git log`.

## What is an ADR?

An **Architecture Decision Record** is a short document (usually one page) that captures one decision:

- the **context** that forced it,
- the **decision** that was made, and
- the **consequences** — what becomes easier, what becomes harder, what's locked in.

The format was introduced by Michael Nygard in his 2011 post [*Documenting Architecture Decisions*](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

## How ADRs work here

- **Numbered sequentially** — `0001-`, `0002-`, ... The number never changes.
- **Append-only** — once an ADR is *Accepted*, do not edit its content. Fix typos and update links, but don't rewrite the decision.
- **Superseded, not deleted** — if a decision is reversed, write a new ADR that *supersedes* the old one, and mark the old one's status as *Superseded by ADR-NNNN*. Both records remain visible.
- **Short** — if it doesn't fit on a page, it's probably more than one decision.

## When to write a new ADR

Write one when:

- You're choosing between two or more credible approaches and the reason matters.
- A constraint (legal, performance, integration) forces a non-obvious design.
- You're locking in something that will be hard to reverse later (a public API shape, a data format, a dependency boundary).
- You're deliberately *not* doing something the reader might expect.

Don't write one for routine refactors, bug fixes, or stylistic preferences — those belong in commit messages.


## Index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-v4-pipeline.md) | V4 pipeline replaces `create_typical_periods` | Accepted |
