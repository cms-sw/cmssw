# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Branch performance-plot validation: the truth-graph producers, the Branch<->reco
# association maps, and the DQM analyzers that turn them into plots comparing the
# truth::Branch graph to the legacy truth objects. Harvesting (efficiency) lives in
# truthGraphDQMHarvester_cff. Hooked into globalValidation behind enableTruth.

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# The logical graph and hit index are built at DIGI (mixing accumulator chain) under
# enableTruth and arrive at RECO through the input. The associators/validators below
# consume them by DetId via string InputTags, so the signal-only build producers are
# intentionally NOT imported here: importing them would attach them to the RECO
# process and shadow the DIGI-built products.

branchHGCalValidator = DQMEDAnalyzer(
    "BranchHGCalValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("mix"),  # merged raw graph, built at DIGI by the accumulator
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    caloParticles=cms.InputTag("mix", "MergedCaloTruth"),
    simClusters=cms.InputTag("mix", "MergedCaloTruth"),
    folder=cms.string("HGCAL/BranchValidator"),
    minPt=cms.double(1.0),
    maxEta=cms.double(3.0),
)

# Tracker counterpart. A TrackingParticle has no hits of its own, so the
# Branch<->TrackingParticle comparison is mediated by the reco track: the
# association producer matches reco tracks to branches by shared tracker simhits,
# and the validator closes the loop to the TrackingParticle via ClusterTPAssociation.
# Phase-2 tracker: pixel + outer-tracker (Phase2TrackerCluster1D), no strips.
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer as _tpClusterProducer
truthTpClusterProducer = _tpClusterProducer.clone(
    pixelClusterSrc=cms.InputTag("siPixelClusters"),
    phase2OTClusterSrc=cms.InputTag("siPhase2Clusters"),
    pixelSimLinkSrc=cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc=cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc=cms.InputTag("mix", "MergedTrackTruth"),
    throwOnMissingCollections=cms.bool(False),
)

branchTrackingValidator = DQMEDAnalyzer(
    "BranchTrackingValidator",
    src=cms.InputTag("truthLogicalGraphProducer"),
    rawSrc=cms.InputTag("mix"),  # merged raw graph, built at DIGI by the accumulator
    hitIndex=cms.InputTag("truthLogicalGraphHitIndexProducer"),
    tracks=cms.InputTag("generalTracks"),
    clusterTPMap=cms.InputTag("truthTpClusterProducer"),
    folder=cms.string("Tracking/BranchValidator"),
    minPt=cms.double(0.9),
    maxEta=cms.double(3.0),
)

# The truth-graph DQM analyzers that compare the graph to the legacy truth objects:
# CaloParticle and SimCluster through branchHGCalValidator, TrackingParticle through
# branchTrackingValidator. Split in two so the release wires each into the right place,
# the EDProducers in the prevalidation Path and the analyzers in the validation EndPath.
# The logical graph and the hit index are built at DIGI by the mixing accumulator chain
# and arrive at RECO through the input file, so the signal-only build producers are not
# imported here: they would attach to the RECO process and shadow the DIGI-built
# products. The validators' rawSrc points at the mixed graph.
truthGraphValidationProducers = cms.Sequence(truthTpClusterProducer)
truthGraphValidationAnalyzers = cms.Sequence(
    branchHGCalValidator
    + branchTrackingValidator
)
