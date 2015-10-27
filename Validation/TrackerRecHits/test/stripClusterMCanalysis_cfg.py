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
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/06D2F696-B336-E511-A10D-0025905964BE.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/4270DF62-B136-E511-8127-0026189438BF.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/B099AF99-B336-E511-90DC-0025905938D4.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/B6BE6E09-B936-E511-8B8C-00261894395C.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/C8C7DCF2-AF36-E511-A978-00261894397B.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/D8974F6E-B936-E511-895D-00261894387C.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/E8C61CCF-B036-E511-962E-0025905A6094.root',
'/store/relval/CMSSW_7_6_0_pre2/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/75X_mcRun2_asymptotic_v2-v1/00000/F40D56A9-B836-E511-8513-0025905A60C6.root' ] );

process = cms.Process("makeNtuples")

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
