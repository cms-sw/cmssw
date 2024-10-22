# Imports
import FWCore.ParameterSet.Config as cms
import os 

# Create a new CMS process
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('assocTest',Run2_2017)

# Import all the necessary files
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

### "Run2"
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# (See /Configuration/AlCa/python/autoCond.py)
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
# Process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Input source
process.source = cms.Source('PoolSource',
  fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_9_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/409345CD-F79C-E711-8383-0CC47A4D76C8.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/9C783844-F79C-E711-96E3-0CC47A4D76AA.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/DC94E5C6-F79C-E711-B3EC-0CC47A4D7606.root'

# '/store/relval/CMSSW_9_0_0/RelValSingleMuPt100_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v5-v1/00000/00C6B64E-4B0F-E711-B45F-0CC47A7C3410.root',
# '/store/relval/CMSSW_9_0_0/RelValSingleMuPt100_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v5-v1/00000/52E12C5D-4A0F-E711-8ACE-0025905A6104.root' 

        #'file:/batch/handies/testFile.root'
        #'file:/tmp/emiglior/step3.root'
        #'file:'+os.environ.get('REMOTEDIR')+'/step3.root'
  )
)

# Output
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:Run2_Trk_rechits_validation.root')
)

process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)


# RecHits are not persistent... re-create them on-the-fly
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")

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

# To enable debugging:
# [scram b clean ;] scram b USER_CXXFLAGS="-DEDM_ML_DEBUG"

# process.load("SimTracker.TrackerHitAssociation.test.messageLoggerDebug_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.TrackAssociator = dict()

# Number of events (-1 = all)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

# Processes to run

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)

# process.validation_step = cms.Path(process.content*process.testassociator)
process.validation_step = cms.Path(process.testassociator)

process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.validation_step)
