# Security policy

## Supported versions

Security fixes are made for the latest release (and `main`). TreeEngine is pre-1.0; upgrade to
the newest version to receive fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub's
"Report a vulnerability" button on the repository's Security tab
(private vulnerability reporting). Include the version, how to reproduce it and the impact.
You will get an acknowledgement within a week; a fix and an advisory follow as soon as possible,
crediting you unless you prefer otherwise.

## Scope

In scope: the library, `treeengine-mcp`, `treeengine-server` (path escapes from the ingest
roots, authentication bypass, unbounded resource use despite the limits, information leaks in
error responses, unsafe parsing of documents).

Out of scope: deployments that disable the protections on purpose (`--insecure`, ingest roots
covering the whole file system), denial of service by a client you authorised, and
vulnerabilities in optional third-party dependencies (report those upstream).

Deployment guidance: [docs/operations/security.md](docs/operations/security.md).
