# Imports
import FWCore.ParameterSet.Config as cms
import os 

# Create a new CMS process
from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process('assocTest',Phase2)

# Import all the necessary files
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

### "Run2"
# process.load('Configuration.StandardSequences.GeometryRecoDB_cff')

### uncomment next fragment for "D17" 
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
# temporary: use fake conditions for LA 
process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkTilted4025_cff')
### end of "D17"

process.load('Configuration.StandardSequences.MagneticField_cff')

process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# (See /Configuration/AlCa/python/autoCond.py)
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-RECO/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/1C1AFFEE-429B-E711-9127-0CC47A7C345E.root',
'/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-RECO/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/362A31F1-3E9B-E711-8BCB-0025905B8598.root',
'/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-RECO/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/3C136CC9-4B9B-E711-A423-0025905AA9F0.root',
'/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-RECO/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/8E16441D-489B-E711-B6A7-0025905B8576.root',
'/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-RECO/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/A81F36AA-469B-E711-B45A-0CC47A4C8ECA.root'
    )
#     , secondaryFileNames = cms.untracked.vstring(
# '/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/14F04A4A-3C9B-E711-9509-0CC47A4D764A.root',
# '/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/18A33AC5-389B-E711-A7EA-0025905A60CE.root',
# '/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/5C456E4D-3C9B-E711-8E08-0025905B8594.root',
# '/store/relval/CMSSW_9_3_0/RelValSingleMuPt100Extended/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/C88F4190-339B-E711-B773-0025905B858A.root'
#     )
)

# Output
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:phase2Trk_rechits_validation.root')
)

process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)


# RecHits are not persistent... re-create them on-the-fly
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')
process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi')
process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi')
# process.siPhase2RecHits.Phase2StripCPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE")

# Insert this in path to see what products the event contains
process.content = cms.EDAnalyzer("EventContentAnalyzer")

# Analyzer
process.testassociator = cms.EDAnalyzer("TestAssociator",
   siPixelRecHits = cms.InputTag("siPixelRecHits"),
   matchedRecHit = cms.InputTag("siStripMatchedRecHits", "matchedRecHit"),
   rphiRecHit = cms.InputTag("siStripMatchedRecHits", "rphiRecHit"),
   stereoRecHit = cms.InputTag("siStripMatchedRecHits", "stereoRecHit"),
   siPhase2RecHits = cms.InputTag("siPhase2RecHits"),
   ### for using track hit association
   #
   associateRecoTracks = cms.bool(False),
   associateHitbySimTrack = cms.bool(False),
   associatePixel = cms.bool(True),       
   associateStrip = cms.bool(True),
   usePhase2Tracker = cms.bool(False),
   pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
   stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
   phase2TrackerSimLinkSrc = cms.InputTag("simSiPixelDigis", "Tracker"),
   ROUList = cms.vstring('TrackerHitsPixelBarrelLowTof',
                         'TrackerHitsPixelBarrelHighTof',
                         'TrackerHitsPixelEndcapLowTof',
                         'TrackerHitsPixelEndcapHighTof',
                         'TrackerHitsTIBLowTof',
                         'TrackerHitsTIBHighTof',
                         'TrackerHitsTIDLowTof',
                         'TrackerHitsTIDHighTof',
                         'TrackerHitsTOBLowTof',
                         'TrackerHitsTOBHighTof',
                         'TrackerHitsTECLowTof',
                         'TrackerHitsTECHighTof')
)
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(process.testassociator,
   usePhase2Tracker = cms.bool(True),
   siPhase2RecHits = cms.InputTag("siPhase2RecHits"),
   pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
   phase2TrackerSimLinkSrc = cms.InputTag("simSiPixelDigis", "Tracker"),
)

# To enable debugging:
# [scram b clean ;] scram b USER_CXXFLAGS="-DEDM_ML_DEBUG"

# process.load("SimTracker.TrackerHitAssociation.test.messageLoggerDebug_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.TrackAssociator = dict()

# Number of events (-1 = all)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# Processes to run

process.rechits_step = cms.Path(process.siPixelRecHits*process.siPhase2Clusters*process.siPhase2RecHits)

# process.validation_step = cms.Path(process.content*process.testassociator)
process.validation_step = cms.Path(process.testassociator)

process.schedule = cms.Schedule(process.rechits_step, process.validation_step)
