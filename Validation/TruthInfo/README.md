# Validation/TruthInfo

DQM validation of the MC-truth graph and of its association maps
(`SimGeneral/TruthGraphAssociatorProducers`).

`TruthBranchRecoValidator` is one templated analyzer over tracks, vertices, secondary
vertices and tracksters. The reco-driven metrics are booked per working point, the
truth-driven ones per truth level, and `DQMGenericClient` harvests all of it.
`TruthBranchHistoProducerAlgo` books and fills the histograms.

`BranchHGCalValidator` and `BranchTrackingValidator` are the first-generation
validators. They measure how well a truth branch reproduces the legacy truth objects,
`CaloParticle` and `SimCluster` for the first, `TrackingParticle` for the second.

The Run4 eras carry `enableTruth`, which schedules the graph summary and the two
first-generation validators in the standard validation. Those book 98 monitor elements.

The association performance plots are not part of the default validation. They book
54242 monitor elements, 21.9 MiB of the harvested DQM file on 10 ttbar D127 events, so a
study of the association turns them on with `customiseTruthBranchValidation` from
`SimGeneral/TruthGraphAssociatorProducers`. It fills `truthBranchValidationSequence` in
the validation EndPath and `truthBranchHarvestingSequence` in HARVESTING. The HLT twins
are in `truthBranchHltValidationSequence` and `truthBranchHltHarvestingSequence`, added
by `customiseTruthHltValidation`, because they read HLT collections an offline
reconstruction does not produce.

`scripts/makeTruthValidationPlots.py` renders the harvested DQM into a gallery.

Documentation: <http://cms-truth.docs.cern.ch/validation/>
