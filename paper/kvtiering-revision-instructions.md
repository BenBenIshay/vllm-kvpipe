# Revision instructions — "Heterogeneous KV Memory for Capacity-Bound LLM Decode"

Scope: claims, structure, storytelling, wording, consistency. **Not** "run more experiments" — pending GH200 slots are assumed to be in progress. Source reviewed: `Desktop\Master\KVPipelining paper\tex\` (2026-09-09 copy).

Priority: **P0** = a reviewer can reject on this alone · **P1** = weakens the paper materially · **P2** = polish.
Each item: *where* → *what is wrong* → *what to do*.

---

## 0. Global rules for the writing agent

- **G1 (P0) Never state an unrun result in the indicative.** Until data exists, write the *prediction* ("the model predicts X; §6.y tests it") — never "§6 confirms", "§6 reports it reaching opposite verdicts", "§6 measures that composition". Grep for `confirms`, `reports`, `measures`, `finds`, `validated` and check each against a `% src:` line. Known offenders: `04-design:379`, `04-design:433`, `07-related:38`, `07-related:131`, `01-intro:105`.
- **G2 (P0) Every projected number must be labelled as projected** in the table header or caption, in the intro as well as §3. Tables 1 and 2 are outputs of `killer_v2.py`, not measurements.
- **G3 (P1) Use the word "validated" only with its object.** "Eq. 1's step-time structure is validated on A100" is true. "The capacity model is validated on hardware" is not (yet). Same for "predicts" vs "is consistent with".
- **G4 (P1) Remove the meta-voice.** The draft repeatedly tells the reader how honest/precise it is being. Delete the framing clause and keep the fact. Examples to rewrite: "we state it as a finding rather than apologise for it", "We do not wave this away", "and we say so deliberately", "Finally, the honest attribution", "It is worth being precise about…", "The structure is worth reading carefully", "The model earns its place by…", "A design rule that accounts for when the technique fails is worth more than one tuned to its successes", "is worth as much as the positive result, and we should say so".
- **G5 (P1) Remove the lab-notebook voice.** The paper narrates the authors' debugging history ("found only by testing", "Two corrections belong in the text because both were ours", "our first reading of it was wrong", "What the transfer path cost us", "we retracted"). A paper reports what is true and, where useful, a *lesson*; it does not narrate how the authors got there. Convert to lessons (see S5) or delete.
- **G6 (P1) One vocabulary.** The intro/§4 say *host-tier / device-tier*; §2, §3, §5, §6 say *rider / resident*; §3's title says *offload*. Pick host-tier/device-tier everywhere. If "rider" is kept as a nickname, define it once in §1 and never use it before that (it currently appears in §2.3 before its definition in §3.1).
- **G7 (P1) One notation.** See section N below. The main.tex header promises symbols cannot drift; they have.

---

## 1. Storyline-level problems (fix these before line edits)

### ST1 (P0) The model and the system tell opposite stories about whether placement matters
- *Where:* §1 contributions 3–4, §3(a), §4.5 Placement.
- *Problem:* §3 proves that freed HBM and link cost both scale with S·κ, i.e. *which* request goes to the host is irrelevant to first order — any policy that fills both byte budgets achieves the full gain. The intro then sells "per-request placement" as a headline contribution. A reviewer will conclude the policy is vacuous by the paper's own algebra.
- *Do:* Add an explicit bridge at the start of §4.5 (and one sentence in §1): **the model fixes *how many* bytes each tier can carry; the manager decides *which* requests and *when*, and these matter only through effects the fluid model omits** — list them concretely (request lifetime/turnover, migration cost, per-request overhead, integer effects, growth during decode, avoiding recompute-preemption). Then every policy claim must tie to one of those listed effects. If no such effect can be argued for a component, downscope that component from "contribution" to "design choice".

### ST2 (P0) The "temporal" placement signal contradicts the novelty claim
- *Where:* `04-design:381–387` vs `02-background:52–56`, `07-related:95–101`.
- *Problem:* The novelty argument is "existing policies only govern requests that are *not currently decoding*; we govern *actively decoding* ones." The placement signal then offered is "a request that is not currently generating / idle" — exactly the case the paper says swap and hierarchical caching already handle. On a decode node under continuous batching every admitted request generates every step, so "idle" is undefined. "Near the end of its generation" is also unexplained: why is a nearly finished request a *better* host-tier candidate?
- *Do:* Define what "idle" means in the system model (tool-call pause? multi-turn gap?) and state whether such requests are in scope. Give the one-sentence causal argument for each signal (what cost comes apart from what benefit, in bytes or time). Reconcile with the novelty paragraph — either narrow the novelty claim or drop the idle signal.

### ST3 (P0) "Every existing system is static and total" is contradicted by the paper's own related work
- *Where:* `01-intro:47–55`, abstract plan item (2) in `00-abstract:6`, `07-related:87–88` ("Nobody runs the mixture"), Table 1 framing.
- *Problem:* Pie (which §7 describes accurately) keeps part of every request's KV in HBM and part on host, with an adaptively tuned split. It is a dynamic, partial, mixture system that uses both bandwidths at once. The intro's dichotomy is therefore false, and the abstract is planned around the same false sentence.
- *Do:* Reframe the gap as **granularity**, not existence: "systems either place nothing, or place a uniform fraction of *every* request; none decides per request." Mention Pie in the intro paragraph, not only in §7. Change "Nobody runs the mixture" accordingly.

### ST4 (P0) The paper never argues why the request axis is better than the layer axis
- *Where:* `07-related:45–70`.
- *Problem:* In the paper's own fluid model a layer split with the right m/n fills the same two byte budgets and reaches the same interior optimum. "Orthogonal, composes" is asserted, but the two approaches spend the *same* link budget — they are substitutes. The layer axis also has an advantage the paper hides: it has no integer quantization at long context (see C4).
- *Do:* Write an explicit paragraph "Why per request?" with arguments that are actually true of the design (e.g. a host-tier request holds zero device blocks so admission needs no allocation; movement happens once per request at controlled moments; only selected requests pay link traffic) **and** concede where layer-split wins. Replace "orthogonal rather than competing" with an accurate statement of how they would share the link budget.

### ST5 (P0) The model omits the cost the paper's own measurement says is first-order
- *Where:* §3(c), Table 2, §4.7, §6.3, `05-impl:54–59`.
- *Problem (writing, not experiment):* §6.3 reports a fixed 12.29 ms per-step cost for using the host path at all. §3's gain law and Table 2's "tput" column (computed with `stage=0`) do not contain it; on a 34 ms GH200 step it would cancel most of the projected ×1.5. The narrative therefore presents a headline projection that the paper's own later section undermines, and the reader discovers this alone.
- *Do:* Put the fixed cost *into the model section*: give the throughput form `gain_tput = (1 + B_off/B_cap) · ITL_0 / (ITL_0 + F + Δ)` and state the resulting **second gate** ("the technique pays only when the capacity gain exceeds the fixed-cost penalty"). This also repairs C7 (the 8B "nothing to buy" contradiction). Delete or qualify "Machine-independent, so it does not corrupt the C2C measurement" — launch overhead depends on the host CPU.

### ST6 (P1) §2.1 describes the wrong regime, and the right one is a better story
- *Where:* `02-background:29–37`.
- *Problem:* (i) "roughly 79% KV read and 21% weight read" contradicts §3(b), where Qwen-32B spends 21 of 34 ms (62%) on *weights*; on A100 the same model is ~79% *weights*. The figure looks inverted or belongs to an unnamed model. (ii) Conceptually, §2.1 dismisses the weight-dominated regime as "not the one a capacity feature should be designed for", yet the gain law's amplifier HBM/C is largest exactly when weights dominate the step.
- *Do:* Name the model/hardware for any split quoted. Rewrite the paragraph around the actual thesis: **the favourable regime is capacity-bound *and* weight-read-heavy — the weight read is a fixed per-step window in which the link is idle, and the larger the weights relative to HBM the longer that window is relative to the resident KV.** This makes §2.1 set up §3 instead of contradicting it.

### ST7 (P1) The scope assumption arrives too late
- *Where:* `01-intro:123–126`.
- *Problem:* Table 1 and the word "additive" appear before the reader learns that everything depends on a P/D-disaggregated decode node. The header comment of §2 itself says a late discovery "reads as concealment".
- *Do:* Move one sentence of scope into the "Two memories, managed as one" paragraph, before Table 1. Keep the full Scope paragraph where it is.

### ST8 (P1) §8.5 tells the reader the paper's mechanism is dominated, without saying why it was built
- *Where:* `08-discussion:48–101`.
- *Problem:* The section concludes that the bounce buffer and D2H mirroring are "a single tax, paid for running on a staged backend" that DirectKV already removes. The natural reviewer question — why not build on the zero-copy path? — is never answered. The section is also long and partly answers an internal question nobody asked the reader ("Nor does it require the host to observe anything… coherence snooping… release-ordered completion flag").
- *Do:* State in one sentence why a staged backend was used (availability, portability to PCIe, etc.) and that results are therefore conservative with respect to mechanism. Cut the "host need not observe" paragraph to one or two sentences or remove it. Target: half the current length.

### ST9 (P1) Tone toward DirectKV
- *Where:* `01-intro:94–99`, `03-model:268–280`, `07-related:92`.
- *Problem:* "explaining a published result its own authors do not", "The paper does not explain the inversion", "does not explain" (×3). The DirectKV authors are likely reviewers. Combined with C6 (the explanation is itself shaky) this is high risk for no gain.
- *Do:* Neutral phrasing: "DirectKV reports X; our model offers an account of it." Say it once, in §3.7, not three times.

---

## 2. Claim-by-claim corrections

### C1 (P0) Contribution 1: "validated on hardware and against a published system's own results"
- *Where:* `01-intro:103–106`, `01-intro:82–84`.
- *Problem:* The 54 A100 configurations validate that ITL is linear in B·S (Eq. 1). They say nothing about the multiplier (Eq. 3) or the gate (Eq. 4). "validate its structure against 54 measured configurations" attaches the evidence to the wrong equation.
- *Do:* "a closed-form multiplier and an applicability gate; the underlying step-time model is validated on A100, and the multiplier is tested in §6." Apply G3.

### C2 (P0) "The model reproduces measurements it was not fitted to … which played no part in constructing it"
- *Where:* `03-model:162–170`.
- *Problem:* η_w = 0.75 and the η_kv table come from the same `p4_*_a100.json` sweeps the prediction is checked against. As worded, the claim is false and easy to catch. Both predictions are also low by 6–7% in the same direction, which reads as a missing term, not noise.
- *Do:* Say exactly what was calibrated on what and what was held out (e.g. "η_w from B=1 points, η_kv from sub-capacity working sets; the full-pool step time is a held-out *configuration*"). If nothing was held out, delete the paragraph. Acknowledge the one-sided residual and name the likely omitted term.

### C3 (P0) §3(b) refutes the approximation that §3(c) then uses
- *Where:* `03-model:71–106`.
- *Problem:* (b) shows ITL_0 ≈ HBM/BW fails by 1.4× across models; (c) substitutes ITL_0 ≈ HBM/(BW·η̄) to obtain the headline law. η̄ then silently disappears from the right-hand side of Eq. 3, and "X/BW ≈ 0.15" only works out as 450/(4000×0.75). `BW` is defined as "achieved" bandwidth in the Setup but multiplied by η as if it were *peak* in Eq. 1.
- *Do:* Present the exact form first (`1 + ITL_0·X/C`), then the approximation with η̄ written explicitly, and state its error for each Table 2 row. Define BW once as peak (and X likewise: spec 450, Pie-measured 419, own measurement pending — the footnote says "we measure rather than assume" while Table 1's 15 GB assumes the full 450).

### C4 (P0) "Context length cancels" / "Context length cannot discriminate"
- *Where:* `03-model:43–59`, `01-intro:85–87`, `04-design:375–379`.
- *Problem:* True only in the fluid limit with uniform S. With whole requests, B_cap and B_off are small integers at long context: Qwen-32B at 32K is 3+1 (×1.33), at 64K 1+0 (×1.0). Table 2's ×1.46 is itself only reproducible with integer floors at 8K, so the table already depends on S. "Long context does *not* help" understates it: long context actively hurts a request-granular design.
- *Do:* Qualify everywhere: "to first order, for uniform context". Add a short paragraph on integer effects and what they imply for request granularity (ties to ST4). In §4.5 change "cannot discriminate … equivalent to ranking arbitrarily" to "does not discriminate to first order".

### C5 (P0) Table 2 is not self-consistent and not reproducible from the text
- *Where:* `03-model:121–145`.
- *Problems:*
  1. Caption: "tput additionally charges the step lengthening" — yet tput > capacity in every row. Since tput = B/ITL, a longer step must make tput ≤ capacity. Either the columns are mislabelled or the definition is different from what is written.
  2. C is defined as HBM − W but every row uses HBM − W − 4 GB (96−64=32, table says 28). The reserve is never mentioned.
  3. The text insists on "measured checkpoint sizes rather than nominal parameter counts" and then uses 16.0 / 32.0 / 64.0 for three of four rows (Qwen2.5-32B is ≈65.5 GB). The paper's own sensitivity paragraph says this error propagates undamped.
  4. "a reader who recomputes Eq. 3 by hand should land on the first column" — only the Qwen-fp16 row can be reproduced, and only with integer floors.
  5. Which number is "the law's prediction"? The precision-pair paragraph quotes the tput column (×1.50→×1.31) while the caption compares Eq. 3 to the capacity column (×1.51 vs ×1.46).
- *Do:* Define each column by formula in the caption; document the reserve and the floors (or remove them and give fluid values); use measured sizes in all rows or drop the sentence; pick one column as "the prediction" and use it consistently in §3, §6.7 and §8.2. Title the table as a projection (G2).

### C6 (P0) "The model explains a published result" (§3.7)
- *Where:* `03-model:265–280`, `01-intro:94–99`, `07-related:89–93`.
- *Problems:*
  1. Eq. 3 compares a *mixture* with all-HBM. DirectKV is *all-host*. At equal latency the law's own all-host/all-HBM ratio is gain − 1 < 1 — it predicts DirectKV loses on *both* models.
  2. DirectKV's OPT win is "where SGLang OOMs". OPT is MHA with ~6–10× the KV bytes/token of Llama-8B; the baseline runs out of memory because of κ·S — the quantity the paper says cancels.
  3. OPT-13B is omitted from the argument: HBM/C ≈ 1.45, close to the 8B's 1.26, yet DirectKV wins on it.
  4. "leaves C ≈ 76 GB of an H100/GH200" is wrong for an 80 GB H100.
  5. "the strongest available evidence that the model captures something real" rests on two qualitative data points chosen post hoc.
- *Do:* Downgrade from "predicts" to "is consistent with", address OPT-13B and the MHA/OOM explanation explicitly, fix the H100 figure, delete "strongest available evidence". Consider shrinking §3.7 to a remark. Remove the claim from the contributions list ("and against a published system's own results").

### C7 (P1) "An 8B model leaves so much HBM free that there is nothing to buy"
- *Where:* `01-intro:89–90`, `03-model:273–275`, `06-eval:141–145`.
- *Problem:* Table 2 projects ×1.19/×1.26 for that model. If +26% is "nothing", the fp8 row's ×1.31 is nothing too; and a measured null on 8B would *falsify* a law predicting 1.26, not confirm it. The "negative control" framing is incoherent as written.
- *Do:* After ST5, define "pays" quantitatively (projected gain vs fixed-cost penalty) and place each model relative to that line. Then the 8B is a coherent predicted-negative.

### C8 (P1) η_kv table vs its own caveat
- *Where:* `03-model:225–255`.
- *Problem:* The caveat says B=16,S=8192 (17.2 GB) gives η_kv = 0.747, but Table 3 lists 0.54–0.56 for 17–34 GB — so the table is built on the point the text calls anomalous, and the clean point is excluded. This suggests η_kv tracks batch size / kernel heuristics rather than bytes, which undercuts the column header "KV working set". The bold sentence "every fixed-bandwidth roofline in this literature mis-predicts…" rests on this table. Also "R² = 0.975 linear in B·S" and "η_kv degrades 1.6× with working set" cannot both be fully true (the second implies convexity).
- *Do:* State which (B,S) points make up each column; present η_kv as a function of what the data actually supports; reconcile linearity with degradation in one sentence; soften the bold generalisation to what was measured (one GPU, one kernel generation). Replace "most likely FlashDecoding's split-KV heuristics" (non-standard name, unverified) with a hedged statement or a profile-backed one. Clarify "54 configurations" = 41 + 13 repeats (are the 13 a subset?).

### C9 (P1) "a state that did not previously exist" / "did not exist before the zero-copy mechanism created it"
- *Where:* `02-background:52–56`, `07-related:100–101`.
- *Problem:* FlexGen, Pie, NEO, OmniServe all decode with host-resident KV; the paper cites them.
- *Do:* Rescope: "no existing system *chooses, per request,* the tier of an actively decoding request."

### C10 (P1) "This is the production default… every major serving stack now supports it"
- *Where:* `02-background:64–67`.
- *Do:* "widely deployed and supported by the major serving stacks"; colocated chunked-prefill serving remains common, and the paper's own negative result says the technique does not apply there — say that in the same paragraph.

### C11 (P1) NEO argument: "improves as the interconnect improves, which is the direction hardware is moving"
- *Where:* `07-related:134–140`; related: `08-discussion:21–32`.
- *Problem:* The paper's own law depends on X/BW and HBM/C, not X. On the next generation HBM bandwidth and capacity both grow faster than C2C, so both factors trend *against* the technique — the same structural decay the paragraph holds against NEO.
- *Do:* Rewrite the NEO comparison in terms of X/BW. Extend the Anti-scaling TODO to cover bandwidth growth, not only capacity growth.

### C12 (P1) Intro Results paragraph: "raise peak concurrency from 8 to 48"
- *Where:* `01-intro:128–131`, Table `tab:capacity`.
- *Problem:* Synthetic KV, no latency reported, and the paper's own admission roofline permits 2 host-tier requests at that context — 48 on PCIe implies ~1 s steps. Presented in the intro as a result, it reads as misleading. The all-HBM column dropping 8→7 as offered load doubles is unexplained. Caption "HBM is no longer the binding constraint" is trivially true once admission ignores the link.
- *Do:* Reframe as a **mechanism check** ("zero-block admission decouples host-tier admission from the device pool"), report the ITL alongside or state plainly that latency is unbounded in this test, and move it out of the intro's headline results. State context length and whether the pool was artificially constrained (the paper criticises CalKV for "synthetically shrunk HBM").

### C13 (P1) Table `tab:ridercost` and the "break-even 3–4 riders" narrative
- *Where:* `06-eval:61–99`.
- *Problems:* tok/s cannot be reconciled with ITL (8 / 49.06 ms = 163 tok/s, not 57.3) — the column is evidently end-to-end over a finite 12-request run. By B/ITL, 3 riders *reduce* steady-state throughput ~10%, and steady-state break-even is r ≈ 8 — essentially the "7 riders" model the text dismisses as wrong. The "measured 3–4" comes from queue drain in a closed finite workload, which is not the throughput-oriented regime the paper targets. "one rider short of parity" inherits the problem. Intro says 78% hidden, table says 79%. The slope also contains the second attention pass, so "fraction of the transfer that hides" is a lower bound, not the quantity named.
- *Do:* Define tok/s; present both the steady-state and the finite-workload break-even and say which regime each belongs to; do not call the queueing-free model "wrong"; fix 78/79; relabel "hidden fraction" as a lower bound; note the fit is 3 points / 2 parameters.

### C14 (P1) Migration probe (§4.7) never says what it measures
- *Where:* `04-design:409–435`, `06-eval:136–139`.
- *Problem:* A demotion always lengthens the step (link traffic + fixed cost + re-read from the slot); its benefit is capacity, which does not show up in ITL. A probe that reverts "if latency regressed" would always revert, and "should flip to beneficial" on GH200 has no mechanism under that metric. Comparing against a recent *minimum* builds in regression-to-the-mean. The section also never says what migration is *for* in steady state.
- *Do:* State the probe's objective metric (goodput? SLO attainment? queue length?) and why a demotion can score as beneficial under it. Justify or drop the min-baseline. Open the subsection with the use cases (promote when device blocks free up; demote instead of recompute-preemption when resident KV grows).

### C15 (P2) Smaller factual fixes
- `06-eval:24` "25.1 GB/s against a 25.1 nominal" — PCIe 4 ×16 nominal is ≈31.5 GB/s.
- `04-design:303` "That ratio, not the interconnect, is what converts host DRAM into device capacity" contradicts §3 (the link budget is the bound). Reword.
- `03-model:69` "cancel exactly" → "cancel" (context grows across the N₂−N₁ tokens).
- `02-background:22`, `03-model:259` — the 116 FLOP/byte balance point is H100; measurements are A100, projections GH200. Name the right device or say "e.g.".
- `07-related:126` ShadowKV reports 6× larger *batch*, not "6× footprint reduction".
- `07-related:25` "warp pipelining" — check spelling against the PDF. "TMA" (DAK) undefined.
- `04-design:150–152` "costs nothing extra" in GEMMs → "negligible while the GEMM is memory-bound"; the paper itself notes an 8B crossing into compute-bound at B≈142.
- `03-model:85–88` "nothing else contends for the interconnect" is fine, but add that H2D writes do consume HBM bandwidth during an HBM-bound step.
- `01-intro:30–32` "hundreds of gigabytes of host DRAM" — note host DRAM bandwidth (the 480 GB Grace config is below C2C's 450 GB/s), or the reader will assume the link is always the bottleneck.
- `07-related:170` "Three of the papers above name that layer as out of scope" — name them.
- "agentic pipelines" listed as throughput-oriented (`03-model:210`) — many are latency-sensitive; justify or drop.

---

## 3. Structure, per section

### S0 Abstract (`00-abstract`)
- Item (2) of the plan repeats the false "static and total in every published system" (ST3). Fix the plan before writing.
- Item (5) "the number": commit now to it being a *measured* number stated with its regime (model, precision, SLO), with the projection clearly separate.

### S1 Introduction
- Order should be: problem → second memory → **scope sentence** (ST7) → gap stated as granularity (ST3) → Table 1 *labelled as projection* → model insight → system → contributions → results.
- Table 1: label as model projection; explain 13 + 7 → 19 (staging cost) since the sentence above says "bounded by the sum"; the "all-host = 7" row borrows the mixture's 34 ms step although an all-host step has no HBM KV read — state the SLO/step assumption or the row looks arbitrary; do not attribute a projected "7" to DirectKV as if it were their number.
- "managed as a single KV memory" — the design is two pools with per-request placement, not one address space. Prefer "jointly managed".
- Contributions: apply C1, C4 ("to first order"), C6 (drop "against a published system's own results"), and label contribution 5's baseline as an emulation of uniform offload on the same staged substrate, not "DirectKV".
- The C2C footnote is duplicated verbatim in §1 and §2 — keep one.

### S2 Background
- Rewrite §2.1 per ST6. Fix the novelty sentence per C9. Soften §2.3 per C10.
- §2.3 uses "riders" before definition (G6).
- The promised negative result ("we report that negative result in §6.2") currently points at a TODO; keep the promise only if the text will exist.

### S3 Model — reorder
Current order makes the reader use parameters before they are introduced (§3(b), (c) cite §3.6) and makes (c) depend on an approximation (b) just refuted. Proposed order:
1. Setup and step-time model (Eq. 1) + its A100 validation.
2. Measured parameters (η_w, η_kv) — moved up.
3. Host-tier budget and first-order cancellation (with C4 caveats).
4. Gain law: exact form → approximation with explicit η̄ (C3).
5. **What the model omits** (new, short): fixed per-step cost and the second gate (ST5), rider KV re-read from HBM, HBM write contention, host-DRAM bandwidth, integer effects, compute term at large B, MoE ("reads every weight once" is false for MoE, yet §8 invokes MoE as the favourable trend).
6. Applicability gate(s).
7. Projection table (C5) + sensitivity.
8. DirectKV remark (C6), shortened.
- Replace the opening "derives four closed-form results, reports which of them survived contact with hardware" — (b) is a refuted simplification, not a result; the sentence sets up a structure the section does not follow.
- Retitle: "When does host-KV *offload* pay?" conflicts with the memory-system framing (G6).

### S4 Design
- The figures are good; keep them. Add one short "what a host-tier request costs" paragraph that connects to §3.5-new: KV crosses the link *and* is then read from the slot in HBM, plus the fixed launch cost.
- State things a systems reviewer will look for and currently cannot find: CUDA-graph/eager status of both arms; what the "single key" is for a request holding zero device blocks; how the rolling region is sized/reserved when W changes per step; why k = 4; fairness/starvation consequences of queue bypass.
- §4.5–4.7: apply ST1, ST2, C14. "We keep the rule a transparent threshold so that its behaviour can be explained" — then state the rule.

### S5 Implementation
- Keep 5.1–5.4 to a paragraph each; 5.1 duplicates §4.3's aliasing argument — say it once.
- Convert "What the transfer path cost us, and why" into **"Lessons from the transfer path"**: three generalisable lessons (layout determines whether DMA can do the gather; hoist schedule-invariant work out of the per-layer path; time async GPU work with operator-level diffs, not phase timers). Drop the chronology and the before/after of the authors' own bugs except one quantified example per lesson (G5).
- Remove or qualify "machine-independent" (ST5).

### S6 Evaluation — reorder around the claims
Current order (capacity → cost → crossover → endpoints → ablation → negative control → precision → correctness → goodput) wanders. Proposed:
1. Setup (incl. the correctness/bf16 paragraph, workload, SLO definition, **definition of goodput**, eager/graph mode, what "all-host" baseline actually is).
2. Does the model hold? — precision pair, predicted-negative (8B), below-gate SLO point. State each prediction *before* its result.
3. Is the host tier additive? — mechanism check (C12).
4. What does a host-tier request cost? — decomposition and crossover (C13).
5. Three-way comparison at matched SLO.
6. Is the policy load-bearing? — ablations tied to ST1's listed effects.
7. End-to-end goodput.
- Each subsection should open with: question → claim under test → prediction.
- Dangling references to fix: `\S\ref{sec:background}` for a "two-part gate" whose subsection was deleted; the bf16 near-tie paragraph promised in a comment but absent; `07-related:131` promises a KV-quantization composition measurement that §6 does not contain.

### S7 Related work
- Apply ST3, ST4, ST9, C9, C11.
- Missing and expected by reviewers (writing task): NanoFlow (argues large-batch serving is compute-bound — directly challenges §2.1 and must be engaged); InfiniGen, FastDecode, LayerKV, CachedAttention/AttentionStore; Llumnix (cross-instance request migration as the alternative to tiering); Infinite-LLM/DistAttention, LoongServe; vAttention/Jenga; vLLM's native swap/CPU-offload and LMCache/HiCache named explicitly rather than as "hierarchical caching"; output-length prediction (needed for "estimated remaining generation"); KV-shrinking architectures (MLA, sliding-window/hybrid attention) and their effect on whether the gate ever opens; tensor parallelism / adding a GPU as the practical alternative.
- Table 4: caption says "systems that share our hardware assumption" — fine, but then say why OmniServe/NEO are excluded.

### S8 Discussion
- Apply ST8, C11.
- Limitations must include what the paper currently leaves unsaid: single-GPU only (no TP); dense GQA models only (no MoE/MLA); uniform-context model; host-DRAM bandwidth and pinned-memory limits; the favourable configurations (64 GB of weights on a 96 GB part) are ones operators often avoid via fp8 or TP — argue why the regime is nonetheless real; generality beyond NVLink-C2C (CXL, other vendors' coherent links).

---

## 4. Notation and terminology (N)

- **N1** `W` = weight bytes (§3) **and** rolling-region blocks (§4 Table 4). Rename one.
- **N2** `k` = KV bytes/token (§3) **and** rolling slots (§4); §4 uses `κ` for bytes/token. Use κ in §3 too.
- **N3** `08-discussion:57` "slot L mod k" uses L as a layer index; L is the layer count elsewhere (ℓ is the index).
- **N4** Table 4's caption claims to cover §3's notation; it does not.
- **N5** `BW` "achieved" vs peak (C3). `X` likewise.
- **N6** "SMs" (`02-background:25`, `04-design:107,155`) undefined — spell out or replace.
- **N7** "FlashDecoding" → the actual kernel/API name. Qwen-32B vs Qwen2.5-32B, Llama-8B vs Llama-3.1-8B: pick the full name at first use, short form after.
- **N8** `b_hbm_max`, `steady_concurrency`, `seqused_k` — internal identifiers; introduce in prose or remove.

---

## 5. Citations and submission hygiene (H)

- **H1 (P0)** `\usepackage[accepted]{mlsys2027}` is the camera-ready option and de-anonymises; the submission needs the blind mode and the MLSys `.bst`, not `plain`.
- **H2 (P0)** Identity leaks in source comments: "(Ben)" in `01-intro:4`, `/mnt/beegfs/benishay/…` in `refs.bib`. Strip before any source upload.
- **H3 (P1)** `\draftfalse` + surviving `\todo`/`\needsgh` = the file does not compile. By design, but it belongs on the checklist.
- **H4 (P1)** Seven bib entries have no authors. `calkv`: the title appears paraphrased — I believe the published title is "Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System"; verify. Note that title collides with this paper's new "Heterogeneous KV Memory" framing — differentiate explicitly in §7 and reconsider the title. `omniserve` and `bottlenecks` titles also look paraphrased; "OmniServe" is also the name of MIT HAN Lab's serving stack — verify which paper is meant.
- **H5 (P1)** Mentioned but never cited: vLLM, SGLang, FlashAttention, Quest, H2O, OPT, Llama, Qwen. Present in bib but never cited: `vllm`, `flashattention2`, `flashattention3`, `gracehopper`.
- **H6 (P1)** Every DirectKV quote and §/Alg. pointer ("§6", "Alg. 3", "§8.1") carries the endpoint argument; re-verify each against the final OSDI PDF, in particular that no hybrid / hot-block-on-GPU mode exists, since "100% on the host, for every request, always" is the premise of ST3's replacement framing too.
