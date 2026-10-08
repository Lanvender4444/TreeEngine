# Platform Engineering Handbook

This handbook collects the operational standards of the platform team.

## Networking

Standards for the networking area.

### Firewall

Overview of firewall.

#### Inbound rules

The inbound rules standard for firewall in the networking area is policy HB-111. Teams must review it every 30 days and record approvals in ticket queue Q111.

#### Outbound rules

The outbound rules standard for firewall in the networking area is policy HB-112. Teams must review it every 31 days and record approvals in ticket queue Q112.

#### Audit logging

The audit logging standard for firewall in the networking area is policy HB-113. Teams must review it every 32 days and record approvals in ticket queue Q113.

### DNS

Overview of dns.

#### Internal zones

The internal zones standard for dns in the networking area is policy HB-121. Teams must review it every 33 days and record approvals in ticket queue Q121.

#### Caching resolvers

The caching resolvers standard for dns in the networking area is policy HB-122. Teams must review it every 34 days and record approvals in ticket queue Q122.

#### DNSSEC

The dnssec standard for dns in the networking area is policy HB-123. Teams must review it every 35 days and record approvals in ticket queue Q123.

### Load balancing

Overview of load balancing.

#### Health checks

The health checks standard for load balancing in the networking area is policy HB-131. Teams must review it every 36 days and record approvals in ticket queue Q131.

#### Sticky sessions

The sticky sessions standard for load balancing in the networking area is policy HB-132. Teams must review it every 37 days and record approvals in ticket queue Q132.

#### TLS termination

The tls termination standard for load balancing in the networking area is policy HB-133. Teams must review it every 38 days and record approvals in ticket queue Q133.

## Storage

Standards for the storage area.

### Block volumes

Overview of block volumes.

#### Snapshots

The snapshots standard for block volumes in the storage area is policy HB-211. Teams must review it every 40 days and record approvals in ticket queue Q211.

#### Encryption at rest

The encryption at rest standard for block volumes in the storage area is policy HB-212. Teams must review it every 41 days and record approvals in ticket queue Q212.

#### Resizing

The resizing standard for block volumes in the storage area is policy HB-213. Teams must review it every 42 days and record approvals in ticket queue Q213.

### Object storage

Overview of object storage.

#### Lifecycle policies

The lifecycle policies standard for object storage in the storage area is policy HB-221. Teams must review it every 43 days and record approvals in ticket queue Q221.

#### Versioning

The versioning standard for object storage in the storage area is policy HB-222. Teams must review it every 44 days and record approvals in ticket queue Q222.

#### Presigned URLs

The presigned urls standard for object storage in the storage area is policy HB-223. Teams must review it every 45 days and record approvals in ticket queue Q223.

### Backups

Overview of backups.

#### Retention

The retention standard for backups in the storage area is policy HB-231. Teams must review it every 46 days and record approvals in ticket queue Q231.

#### Restore drills

The restore drills standard for backups in the storage area is policy HB-232. Teams must review it every 47 days and record approvals in ticket queue Q232.

#### Offsite copies

The offsite copies standard for backups in the storage area is policy HB-233. Teams must review it every 48 days and record approvals in ticket queue Q233.

## Security

Standards for the security area.

### Identity

Overview of identity.

#### Single sign-on

The single sign-on standard for identity in the security area is policy HB-311. Teams must review it every 50 days and record approvals in ticket queue Q311.

#### Service accounts

The service accounts standard for identity in the security area is policy HB-312. Teams must review it every 51 days and record approvals in ticket queue Q312.

#### Password policy

The password policy standard for identity in the security area is policy HB-313. Teams must review it every 52 days and record approvals in ticket queue Q313.

### Secrets

Overview of secrets.

#### Vault usage

The vault usage standard for secrets in the security area is policy HB-321. Teams must review it every 53 days and record approvals in ticket queue Q321.

#### Rotation

The rotation standard for secrets in the security area is policy HB-322. Teams must review it every 54 days and record approvals in ticket queue Q322.

#### Break-glass access

The break-glass access standard for secrets in the security area is policy HB-323. Teams must review it every 55 days and record approvals in ticket queue Q323.

### Incident response

Overview of incident response.

#### Severity levels

The severity levels standard for incident response in the security area is policy HB-331. Teams must review it every 56 days and record approvals in ticket queue Q331.

#### On-call escalation

The on-call escalation standard for incident response in the security area is policy HB-332. Teams must review it every 57 days and record approvals in ticket queue Q332.

#### Postmortems

The postmortems standard for incident response in the security area is policy HB-333. Teams must review it every 58 days and record approvals in ticket queue Q333.

## Observability

Standards for the observability area.

### Metrics

Overview of metrics.

#### Retention windows

The retention windows standard for metrics in the observability area is policy HB-411. Teams must review it every 60 days and record approvals in ticket queue Q411.

#### Cardinality limits

The cardinality limits standard for metrics in the observability area is policy HB-412. Teams must review it every 61 days and record approvals in ticket queue Q412.

#### Dashboards

The dashboards standard for metrics in the observability area is policy HB-413. Teams must review it every 62 days and record approvals in ticket queue Q413.

### Logging

Overview of logging.

#### Log levels

The log levels standard for logging in the observability area is policy HB-421. Teams must review it every 63 days and record approvals in ticket queue Q421.

#### PII scrubbing

The pii scrubbing standard for logging in the observability area is policy HB-422. Teams must review it every 64 days and record approvals in ticket queue Q422.

#### Shipping agents

The shipping agents standard for logging in the observability area is policy HB-423. Teams must review it every 65 days and record approvals in ticket queue Q423.

### Tracing

Overview of tracing.

#### Sampling

The sampling standard for tracing in the observability area is policy HB-431. Teams must review it every 66 days and record approvals in ticket queue Q431.

#### Span attributes

The span attributes standard for tracing in the observability area is policy HB-432. Teams must review it every 67 days and record approvals in ticket queue Q432.

#### Trace storage

The trace storage standard for tracing in the observability area is policy HB-433. Teams must review it every 68 days and record approvals in ticket queue Q433.

## Deployment

Standards for the deployment area.

### CI pipeline

Overview of ci pipeline.

#### Build cache

The build cache standard for ci pipeline in the deployment area is policy HB-511. Teams must review it every 70 days and record approvals in ticket queue Q511.

#### Test sharding

The test sharding standard for ci pipeline in the deployment area is policy HB-512. Teams must review it every 71 days and record approvals in ticket queue Q512.

#### Artifact signing

The artifact signing standard for ci pipeline in the deployment area is policy HB-513. Teams must review it every 72 days and record approvals in ticket queue Q513.

### Release process

Overview of release process.

#### Canary releases

The canary releases standard for release process in the deployment area is policy HB-521. Teams must review it every 73 days and record approvals in ticket queue Q521.

#### Rollback

The rollback standard for release process in the deployment area is policy HB-522. Teams must review it every 74 days and record approvals in ticket queue Q522.

#### Change freeze

The change freeze standard for release process in the deployment area is policy HB-523. Teams must review it every 75 days and record approvals in ticket queue Q523.

### Environments

Overview of environments.

#### Staging parity

The staging parity standard for environments in the deployment area is policy HB-531. Teams must review it every 76 days and record approvals in ticket queue Q531.

#### Ephemeral previews

The ephemeral previews standard for environments in the deployment area is policy HB-532. Teams must review it every 77 days and record approvals in ticket queue Q532.

#### Production access

The production access standard for environments in the deployment area is policy HB-533. Teams must review it every 78 days and record approvals in ticket queue Q533.

## Databases

Standards for the databases area.

### PostgreSQL

Overview of postgresql.

#### Connection pooling

The connection pooling standard for postgresql in the databases area is policy HB-611. Teams must review it every 80 days and record approvals in ticket queue Q611.

#### Vacuum tuning

The vacuum tuning standard for postgresql in the databases area is policy HB-612. Teams must review it every 81 days and record approvals in ticket queue Q612.

#### Replication lag

The replication lag standard for postgresql in the databases area is policy HB-613. Teams must review it every 82 days and record approvals in ticket queue Q613.

### Redis

Overview of redis.

#### Eviction policy

The eviction policy standard for redis in the databases area is policy HB-621. Teams must review it every 83 days and record approvals in ticket queue Q621.

#### Persistence

The persistence standard for redis in the databases area is policy HB-622. Teams must review it every 84 days and record approvals in ticket queue Q622.

#### Cluster mode

The cluster mode standard for redis in the databases area is policy HB-623. Teams must review it every 85 days and record approvals in ticket queue Q623.

### Migrations

Overview of migrations.

#### Online schema changes

The online schema changes standard for migrations in the databases area is policy HB-631. Teams must review it every 86 days and record approvals in ticket queue Q631.

#### Review checklist

The review checklist standard for migrations in the databases area is policy HB-632. Teams must review it every 87 days and record approvals in ticket queue Q632.

#### Backfills

The backfills standard for migrations in the databases area is policy HB-633. Teams must review it every 88 days and record approvals in ticket queue Q633.
