---
name: google-cloud
description: >
  Expert guidance for designing, deploying, securing, operating, and cost-optimizing
  systems on Google Cloud Platform (GCP). Covers the gcloud/gsutil/bq CLIs, the resource
  hierarchy and IAM, compute (Compute Engine, GKE, Cloud Run, Cloud Functions),
  networking (VPC, load balancing, Cloud NAT, Private Service Connect), data and storage
  (Cloud Storage, BigQuery, Cloud SQL, Spanner, Memorystore, Pub/Sub), Infrastructure-as-
  Code (Terraform, Config Connector), security (service accounts, Workload Identity,
  Secret Manager, org policies), observability (Logging, Monitoring, Trace), and FinOps
  (billing, committed/sustained-use discounts, Recommender, labels). USE THIS SKILL
  whenever the task touches GCP, gcloud, GKE, BigQuery, Cloud Run, Cloud Storage/GCS, VPC,
  IAM service accounts, Terraform on Google Cloud, Google Cloud billing or cost
  optimization, or architecting/migrating on Google Cloud — even when the user only names
  a service (e.g. "deploy a container", "set up a bucket", "query a dataset").
---

# Google Cloud

Operate as a Google Cloud principal engineer. Reason from the resource model and the
billing/security boundaries first, then prescribe concrete `gcloud` commands or
Terraform. Every command is reproducible, every IAM grant is least-privilege, every
recurring resource carries a cost implication that must be stated.

## Operating rules

1. **State the boundary before the command.** Identify the project, region/zone,
   network, and identity (user vs service account) a command runs against. Most GCP
   errors are scope errors, not syntax errors.
2. **Least privilege by default.** Never grant `roles/owner` or `roles/editor` to fix a
   permission error. Find the specific predefined role; if none fits, define a custom
   role. Grant on the narrowest resource (resource > project > folder > org).
3. **Name the cost.** Any resource that runs continuously (VM, load balancer, NAT
   gateway, Cloud SQL instance, idle GKE node pool, provisioned IP, egress) gets an
   explicit cost note. Prefer scale-to-zero (Cloud Run, GKE Autopilot, Cloud Functions)
   when traffic is bursty.
4. **Reproducible, not click-ops.** Prefer `gcloud` over Console steps, and Terraform
   over imperative `gcloud` for anything that must persist or be reviewed. Pin versions.
5. **Verify version-specific facts.** Service limits, machine types, pricing, and GA/
   preview status change. For exact current values, check `cloud.google.com/docs` and
   `gcloud ... --help`; do not assert volatile numbers from memory.

## Resource hierarchy (the spine of IAM and billing)

IAM policies and org policies are inherited downward. A grant at the folder level
applies to every project beneath it. Billing attaches at the project level.

```
                    ┌─────────────────────┐
                    │     Organization    │  ← root; org policies, domain identity
                    └──────────┬──────────┘
                ┌──────────────┼──────────────┐
          ┌─────┴─────┐  ┌─────┴─────┐   ┌─────┴─────┐
          │  Folder   │  │  Folder   │   │  Folder   │  ← e.g. prod / nonprod / shared
          │  (prod)   │  │ (nonprod) │   │ (security)│     IAM + policy inheritance
          └─────┬─────┘  └───────────┘   └───────────┘
        ┌───────┼───────┐
   ┌────┴───┐ ┌──┴────┐ ┌┴──────┐
   │Project │ │Project│ │Project│  ← billing boundary, API enablement, quota, isolation
   └────┬───┘ └───────┘ └───────┘
        │
   ┌────┴────────────────────────────┐
   │ Resources: VMs, buckets, GKE,   │  ← resource-level IAM possible (narrowest grant)
   │ datasets, SQL instances, ...    │
   └─────────────────────────────────┘

Inheritance: Org policy + IAM flow DOWN. Effective access at a resource =
union of grants at resource ∪ project ∪ folder ∪ org.
```

Design implication: separate environments by **project** (hard isolation of quota,
billing, blast radius), group them by **folder** (shared policy), and never co-mingle
prod and nonprod in one project. A dedicated project per environment is also the lever
that unlocks distinct billing labels and lifecycle/retention policy without cross-talk.

## CLI core

Three CLIs cover almost everything; `gcloud` is the primary.

```bash
# Identity & context — always confirm before destructive work
gcloud auth login                       # interactive user creds
gcloud auth application-default login    # ADC for local SDKs/Terraform
gcloud config set project PROJECT_ID
gcloud config set compute/region us-central1
gcloud config list                       # show active account/project/region
gcloud config configurations list        # named context profiles (prod vs dev)

# Projects, APIs, IAM
gcloud projects list
gcloud services enable run.googleapis.com compute.googleapis.com
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SA@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker" --condition=None

# Storage (gcloud storage supersedes gsutil; prefer it)
gcloud storage cp ./file gs://BUCKET/path/
gcloud storage ls -r gs://BUCKET/

# BigQuery
bq query --use_legacy_sql=false 'SELECT 1'
bq ls DATASET
```

Conventions to enforce in generated commands:
- Pass `--project`, `--region`/`--zone` explicitly in scripts (don't rely on ambient
  config that differs per machine).
- Add `--condition=None` on `add-iam-policy-binding` unless you intend a conditional
  grant (otherwise the CLI prompts and scripts hang).
- Use `--format=json` / `--format='value(field)'` for machine-parseable output and
  `--filter` to avoid client-side grep.
- Use `--impersonate-service-account` instead of downloading SA keys.

## Choosing a compute service

```
            Need to run code/containers on GCP?
                          │
        ┌─────────────────┼──────────────────────────┐
        │                 │                           │
  Event/HTTP, short,   Stateless HTTP/         Need full control of OS,
  scale-to-zero?       gRPC container,         GPUs, kernel, sustained
        │              bursty, scale-to-zero?  load, or k8s ecosystem?
        │                 │                           │
        ▼                 ▼                           ▼
  Cloud Functions    Cloud Run            ┌───────────┴───────────┐
  (2nd gen)          (fully managed)      │                       │
                                     Compute Engine            GKE
                                     (VMs, MIGs)         Autopilot | Standard
                                          │                       │
                                  Max control, you      Containers at scale,
                                  patch & scale the      service mesh, operators.
                                  OS. Use MIGs +         Autopilot = no node mgmt,
                                  committed-use          pay per pod. Standard =
                                  discounts for          you size node pools (and
                                  steady workloads.      pay for idle nodes).
```

Heuristics:
- **Cloud Run** is the default for stateless web/API workloads. Scales to zero, pay per
  request + CPU/mem while serving. Lowest ops burden.
- **GKE Autopilot** when you need Kubernetes semantics without managing nodes; you pay
  for requested pod resources, not idle nodes.
- **GKE Standard** when you need DaemonSets, GPUs with custom node config, tight bin-
  packing, or specific node pools. You pay for nodes whether or not pods fill them — an
  idle node pool is a classic silent cost.
- **Compute Engine** for stateful systems, licensed software, GPUs, predictable steady
  load (apply committed-use discounts), or lift-and-shift VMs.
- **Cloud Functions (2nd gen)** for glue/event handlers (Pub/Sub, GCS, HTTP triggers).
  2nd gen runs on Cloud Run + Eventarc under the hood.

## Reference files — read the relevant one before going deep

| Domain | File | Read when the task involves |
|---|---|---|
| Compute & containers | `references/compute.md` | GCE, MIGs, GKE (Autopilot/Standard), Cloud Run, Functions, autoscaling, GPUs |
| Networking | `references/networking.md` | VPC, subnets, firewall, load balancers, Cloud NAT, Private Service Connect, peering, DNS |
| IAM & security | `references/iam-security.md` | service accounts, Workload Identity, roles, org policies, Secret Manager, KMS |
| Data & storage | `references/data-storage.md` | Cloud Storage classes, BigQuery, Cloud SQL, Spanner, Memorystore, Pub/Sub |
| IaC & deployment | `references/iac-deployment.md` | Terraform on GCP, Config Connector, Cloud Build, Artifact Registry, CI/CD |
| Cost & FinOps | `references/cost-finops.md` | billing, CUD/SUD, Recommender, labels, budgets, BigQuery cost, egress |
| Observability | `references/observability.md` | Cloud Logging, Monitoring, Trace, Error Reporting, SLOs, alerting |

Each reference is self-contained and includes commands plus an ASCII diagram. Load only
the file matching the current task; do not preload all of them.

## Common failure modes to preempt

- **Permission denied** → almost always a missing IAM role on the *caller* (user or the
  service's runtime SA), or an API not enabled. Check `gcloud services list --enabled`
  and the effective policy with `gcloud projects get-iam-policy`.
- **Cloud Run / GKE can't pull image** → the runtime service account lacks
  `roles/artifactregistry.reader` on the registry project.
- **VM has no outbound internet but no external IP** → needs **Cloud NAT** on the subnet's
  region; private instances do not egress without it.
- **Cross-project access "works in console, fails in code"** → the service account, not
  your user identity, is the principal at runtime; grant the role to the SA.
- **Surprise bill** → idle GKE node pools, orphaned external IPs (charged when reserved
  and unattached), inter-region/internet egress, BigQuery full-table scans, and forgotten
  Cloud SQL instances. See `references/cost-finops.md`.

## Default deliverable shape

When asked to "set up X", produce, in order: (1) the boundary (project/region/identity
and APIs to enable), (2) the minimal IAM grants, (3) the reproducible `gcloud` or
Terraform, (4) a one-line cost note, (5) the verification command that proves it works.
