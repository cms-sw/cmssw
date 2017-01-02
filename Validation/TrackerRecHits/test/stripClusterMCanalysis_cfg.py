
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
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/346CEC78-33BB-E611-902E-0CC47A4D76AA.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/3A712A49-33BB-E611-BE54-0CC47A4D7628.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/3EB22AC4-36BB-E611-B0C0-0CC47A4D767C.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/40CDA718-34BB-E611-9F98-0CC47A7C3408.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/78040FFD-32BB-E611-ADB5-0CC47A4D762E.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/A8BA78FD-32BB-E611-A850-0CC47A7C34A6.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/AC02816C-33BB-E611-8321-0CC47A7AB7A0.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/C880B8DC-32BB-E611-A77E-0CC47A4D767C.root',
'/store/relval/CMSSW_8_1_0/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/81X_mcRun2_asymptotic_v12-v1/10000/FE9167BA-36BB-E611-9DFB-0CC47A4D76AA.root' ] );

from Configuration.StandardSequences.Eras import eras

process = cms.Process('makeNtuples',eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
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
    *process.InitialStep
    *process.firstStepPrimaryVertices
    *process.StripClusterMCanalysis
    )
