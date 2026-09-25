# Publication implementation snapshot

This is a compact code-only snapshot for auditing the implementation behind the current paper.

Paper task IDs:
20, 21, 22, 23, 24, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42

Included:
- current simulator/runtime code
- the 18 paper scenario implementations and wrappers
- corrected 396-fall campaign/event/split builders
- late-stage lightweight and Protechto CNN evaluation chains
- curated canonical Protechto source preserved in the existing paper evidence archive
- validation and audit scripts

Not included:
- raw KFall or UniVrFall data
- raw simulator sensor CSVs
- checkpoints
- videos
- large result folders

MANIFEST.csv records source path, snapshot path, size, modification time, and SHA-256.

The GitHub audit should establish which exact late-stage evaluation chain generated the manuscript Table VI / Figs. 6-7, after which a tiny local extractor can compute fold-level dataset counts, duplicate simulated draws, and leakage checks.
