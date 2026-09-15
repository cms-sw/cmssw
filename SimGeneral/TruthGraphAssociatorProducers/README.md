# SimGeneral/TruthGraphAssociatorProducers

Association maps between reconstructed objects and truth branches of the MC-truth graph
(`SimDataFormats/TruthInfo`, `PhysicsTools/TruthInfo`).

`TruthBranchTargetsProducer` computes the truth-side targets once per event:
`selectedRoots`, the candidate roots the associators match against; `assignableRoots`,
the subset an adaptive working point may answer with; the signal-seed denominators; and
one truth-to-reco denominator per truth level.

One associator per domain consumes those targets and a reco collection:

| Module | cfi label |
|---|---|
| `AllTrackToTruthBranchAssociatorsProducer` | `allTrackToTruthBranchAssociators` |
| `TruthBranchTracksterAssociatorsProducer` | `truthBranchTracksterAssociators` |
| `AllVertexToTruthBranchAssociatorsProducer` | `allVertexToTruthBranchAssociators` |
| `AllSecondaryVertexToTruthBranchAssociatorsProducer` | `allSecondaryVertexToTruthBranchAssociators` |

Each emits one reco-driven `ticl::TICLAssociationMap` per working point,
`<key>RecoToTruth<WorkingPoint>`, and one truth-driven map, `<key>TruthToReco`. The
collection labels, the working points and the truth levels live in
`python/truthGraphAssociationLabels_cff.py`, so a collection is configured in one place
and `Validation/TruthInfo` reads the same lists.

The Run4 eras carry `enableTruth`, which schedules `truthGraphAssociatorsSequence` in
the prevalidation Path of the standard validation. A job that wants the maps in its own
output applies `customiseTruthGraphAssociators`, which also schedules the HLT twins.

Documentation: <http://cms-truth.docs.cern.ch/association-layer/>
