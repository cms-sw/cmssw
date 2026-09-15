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

The Run4 eras carry `enableTruth`, which schedules `truthBranchValidationSequence` in
the validation EndPath and `truthBranchHarvestingSequence` in HARVESTING. The HLT twins
are in `truthBranchHltValidationSequence` and `truthBranchHltHarvestingSequence`,
because they read HLT collections an offline reconstruction does not produce.

`scripts/makeTruthValidationPlots.py` renders the harvested DQM into a gallery.

Documentation: <http://cms-truth.docs.cern.ch/validation/>
