Subject: GH200 96GB availability + a few infrastructure questions (research workload)

Hi,

We're a research group benchmarking LLM inference on NVIDIA Grace Hopper, and
we're looking to rent a single GH200 for roughly 80 GPU-hours over the coming
weeks. Third-party pricing trackers list Runcrate as offering GH200 96GB at
$2.52/GPU-hour, but I couldn't find it on your own pricing page or in
/dedicated/available-gpus (which lists H100, H200, B200, B300), so my first
question is simply:

0. Do you currently offer NVIDIA GH200 96GB — and if so, on-demand or only as
   reserved/dedicated capacity?

If yes, six things decide whether it fits our experiment. I've tried to explain
why each matters so you can tell us quickly if something is a non-starter.

1. EXCLUSIVITY. Is the GPU exclusive to our instance, or shared / MIG-
   partitioned? Is the instance bare-metal or a VM? Our measurements are
   ratios between two configurations, and a co-tenant on the same device makes
   them meaningless — we've already seen a 9.4x measurement error from
   contention on a shared node.

2. PERSISTENT STORAGE. Your docs mention persistent volumes (/storage/*). We
   need to snapshot ~71 GB (a built Python environment plus a 62 GB model
   checkpoint) so a later rental restores in ~1 hour instead of rebuilding for
   8. What's the throughput restoring a volume of that size, and does a volume
   survive instance termination?

3. IMAGE. Do we get a CUDA *devel* image (one that contains `nvcc`), or
   runtime-only? Can we choose, or bring our own? Not a blocker for us, but
   `nvcc` decides whether two optional experiments can run at all.

4. HOST MEMORY AND PINNED PAGES. How much host RAM does the GH200 instance
   have (we need ≥256 GB), and is there any limit on *pinned* (page-locked)
   memory — a cgroup limit or a low `ulimit -l`? We pin up to ~87 GB of host
   memory, and a pinned-allocation failure typically surfaces as an
   unexplained crash rather than a clear error, so we'd rather know up front.

5. BILLING. On-demand hourly (or per-second) with early release, or a minimum
   committed term? We have a go/no-go decision point about 16 hours in, and
   the ability to stop there is worth more to us than a discount.

6. AVAILABILITY AND REGION. What's realistic availability over the next 4-6
   weeks, and can we get the same physical host (or at least the same region
   and SKU) across two sessions a few days apart? Continuity between sessions
   matters more to us than the hourly rate.

Happy to move this to Slack Connect if that's easier.

Thanks,
Ben
