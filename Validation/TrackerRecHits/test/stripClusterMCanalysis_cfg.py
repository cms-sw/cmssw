
import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
duplCheck = cms.untracked.string('noDuplicateCheck')
source = cms.Source ("PoolSource",
                     fileNames = readFiles,
                     secondaryFileNames = secFiles,
                     skipEvents = cms.untracked.uint32(0),
                     )
readFiles.extend( [
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/00242170-3BC2-E611-AC9F-0CC47A7C3628.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/1EF62F8F-3BC2-E611-B465-0CC47A78A440.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/24E6DCFB-62C2-E611-929B-0CC47A7C35E0.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/2A105509-62C2-E611-B1BC-0CC47A7C3430.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/38366875-3BC2-E611-86EA-0CC47A7C356A.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/6E9E6877-3BC2-E611-B5BC-0025905A60D0.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/DC4B89FA-62C2-E611-BEA8-0025905A6060.root',
'/store/relval/CMSSW_9_0_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v0-v1/10000/EEB73D90-3BC2-E611-B0A4-0025905A60FE.root' ] );


from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('makeNtuples',Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('RecoTracker.TkSeedGenerator.trackerClusterCheck_cfi')
process.load('RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi')
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
process.load('Validation.TrackerRecHits.test.StripClusterMCanalysis_cfi')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.StripClusterMCanalysis.printOut = cms.untracked.int32(0)

useCrossingFrames = False

if useCrossingFrames:
  # Customize mixing module (needed for crossing frames only)
  process.mix.playback = True
  process.mix.digitizers = cms.PSet()
  for a in process.aliases: delattr(process, a)
  process.mix.mixObjects.mixSH.crossingFrames = cms.untracked.vstring(
    'TrackerHitsTECHighTof',
    'TrackerHitsTECLowTof',
    'TrackerHitsTIBHighTof',
    'TrackerHitsTIBLowTof',
    'TrackerHitsTIDHighTof',
    'TrackerHitsTIDLowTof',
    'TrackerHitsTOBHighTof',
    'TrackerHitsTOBLowTof')

  process.StripClusterMCanalysis.ROUList = cms.vstring(
    'g4SimHitsTrackerHitsTIBLowTof',   # crossing frames, module label mix
    'g4SimHitsTrackerHitsTIBHighTof',
    'g4SimHitsTrackerHitsTIDLowTof',
    'g4SimHitsTrackerHitsTIDHighTof',
    'g4SimHitsTrackerHitsTOBLowTof',
    'g4SimHitsTrackerHitsTOBHighTof',
    'g4SimHitsTrackerHitsTECLowTof',
    'g4SimHitsTrackerHitsTECHighTof'
    )

### conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.source = source

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('clusNtuple.root')
)

if useCrossingFrames:
  process.raw2digi_step = cms.Sequence(process.mix*process.RawToDigi)
else:
  process.raw2digi_step = cms.Sequence(process.RawToDigi)

process.p1 = cms.Path(
    process.raw2digi_step
    *process.bunchSpacingProducer
    *process.trackerlocalreco
    *process.offlineBeamSpot
    *process.MeasurementTrackerEventPreSplitting
    *process.siPixelClusterShapeCachePreSplitting
    *process.trackerlocalreco
    *process.calolocalreco
    *process.InitialStepPreSplitting
    *process.trackerClusterCheck
    *process.InitialStep
    *process.firstStepPrimaryVertices
    *process.StripClusterMCanalysis
    )
