"""
This script runs the SimpleTrackValidationEtaBins.
It is just meant for testing and development.

To run:
cmsRun SimpleTrackValidationEtaBins_cfg.py inputFiles=file:/path/to/step2.root
"""

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing ('analysis')
options.inputFiles = ''
options.parseArguments()

process = cms.Process("SimpleTrackValidationEtaBins",Phase2C17I13M9)

# maximum number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(options.inputFiles),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_hltSiPixelRecHits_*_HLTX'   # we will reproduce them to have their local position available
    ),
    secondaryFileNames = cms.untracked.vstring()
)

### conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')

# service to get the root file without passing through the Harvestig
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('simple_validation_etabins.root')
)

process.hltTPClusterProducer = cms.EDProducer("ClusterTPAssociationProducer",
    mightGet = cms.optional.untracked.vstring,
    phase2OTClusterSrc = cms.InputTag("hltSiPhase2Clusters"),
    phase2OTSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker"),
    pixelClusterSrc = cms.InputTag("hltSiPixelClusters"),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis","Pixel"),
    simTrackSrc = cms.InputTag("g4SimHits"),
    stripClusterSrc = cms.InputTag("hltSiStripRawToClustersFacility"),
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    throwOnMissingCollections = cms.bool(True),
    trackingParticleSrc = cms.InputTag("mix","MergedTrackTruth")
)

process.hltTrackAssociatorByHits = cms.EDProducer("QuickTrackAssociatorByHitsProducer",
    AbsoluteNumberOfHits = cms.bool(False),
    Cut_RecoToSim = cms.double(0.75),
    PixelHitWeight = cms.double(1.0),
    Purity_SimToReco = cms.double(0.75),
    Quality_SimToReco = cms.double(0.5),
    SimToRecoDenominator = cms.string('reco'),
    ThreeHitTracksAreSpecial = cms.bool(False),
    UseGrouped = cms.bool(False),
    UseSplitting = cms.bool(False),
    cluster2TPSrc = cms.InputTag("hltTPClusterProducer"),
    useClusterTPAssociation = cms.bool(True)
)

process.SimpleTrackValidationEtaBins = cms.EDAnalyzer("SimpleTrackValidationEtaBins",
    chargedOnlyTP = cms.bool(True),
    intimeOnlyTP = cms.bool(False),
    invertRapidityCutTP = cms.bool(False),
    lipTP = cms.double(30.0),
    maxPhiTP = cms.double(3.2),
    maxRapidityTP = cms.double(2.5),
    minHitTP = cms.int32(0),
    minPhiTP = cms.double(-3.2),
    minRapidityTP = cms.double(-2.5),
    pdgIdTP = cms.vint32(),
    ptMaxTP = cms.double(1e+100),
    ptMinTP = cms.double(0.9),
    signalOnlyTP = cms.bool(True),
    stableOnlyTP = cms.bool(False),
    tipTP = cms.double(3.5),
    trackAssociator = cms.untracked.InputTag("hltTrackAssociatorByHits"),
    trackLabels = cms.VInputTag("hltPhase2PixelTracks"),
    trackingParticles = cms.InputTag("mix","MergedTrackTruth")
)

####  set up the paths
process.simDoubletProduction = cms.Path(
    process.hltTPClusterProducer *
    process.hltTrackAssociatorByHits *
    process.SimpleTrackValidationEtaBins 
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(1),
    wantSummary = cms.untracked.bool(True)
)
