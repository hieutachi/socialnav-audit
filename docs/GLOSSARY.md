# Glossary

- **ung**: unguided/anchor candidate baseline.
- **pt**: point/nominal risk selector.
- **exp**: expected-risk selector over planning futures.
- **cvar**: Conditional Value at Risk / tail-risk selector.
- **gui**: guided candidate pool baseline.
- **guided_rr_***: guided pool plus reranking using selector `*`.
- **ins**: in-sample planning-future score.
- **hel**: held-out contact proxy averaged over state × held-out futures.
- **sprite-conditional CI**: interval formed conditional on the sampled sprites; script 48 uses fixed windows as evaluation subset.
- **window spread**: descriptive variability across fixed windows; not a confidence interval.
- **PICP**: prediction interval coverage probability; not a headline result in this audit.
- **ADE/FDE/MAOE**: average/final displacement error and mean angular error; definitions vary by evaluator. `47b div_pw` is candidate pairwise ADE-like diversity, not accuracy.
- **headroom**: clearance/contact buffer in the evaluated geometry; exact threshold belongs to script protocol.
- **flip rate**: fraction of state decisions whose selected candidate changes between the compared scoring rules.
- **open-loop**: evaluate a proposed path against stored/simulated futures without feeding robot action back into a live environment.
- **oracle-advantaged**: uses information unavailable at decision time (e.g., true future velocity); upper bound only.
- **Flow Matching**: generative vector-field modeling/sampling; this source uses custom scheduler/sampler code, not necessarily a standard DDPM setup.
