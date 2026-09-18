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

Nothing schedules these producers by default, because only a study of the association
reads their maps. `customiseTruthGraphAssociators` schedules them, offline and HLT, and
keeps their products in the output. `customiseTruthBranchValidation` schedules the
offline ones together with the DQM that turns them into performance plots.

Documentation: <http://cms-truth.docs.cern.ch/association-layer/>
