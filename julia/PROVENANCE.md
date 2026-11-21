# Julia provenance

Triangle scripts, research notes and archived results were imported from the
local sibling `triangle_adversary_sdp` project. Changes made during import:
project selection in documentation, default output/store paths to `results/`,
and the SDPLRPlus manifest path to `vendor/SDPLRPlus`.

SymbolicWedderburn is the registered Julia package, pinned at 0.4.4 in the
manifest (tree `e7c6219f8ed7239be4250ff71831dd1413c5951d`). Its upstream is
https://github.com/kalmarek/SymbolicWedderburn.jl . The separate local 0.5.0
checkout is not used by the original triangle environment.

Vendored SDPLRPlus sources came from the local `SDPLRPlus.jl` checkout at
commit `a33d6b4912c1918858232c7b519748ef52b187d3`, including existing local modifications to
`src/SDPLRPlus.jl`, `src/lowrankopt.jl` and `src/sdplr.jl`.
Only its package metadata, README, license and source tree are included.
Its upstream license is preserved at `vendor/SDPLRPlus/LICENSE`.
