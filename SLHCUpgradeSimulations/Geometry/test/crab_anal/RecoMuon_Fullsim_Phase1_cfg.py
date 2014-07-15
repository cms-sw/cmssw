# Auto generated configuration file
# using: 
# Revision: 1.155 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s RECO -n 100 --conditions MC_3XY_V9A::All --datatier GEN-SIM-RECO --eventcontent RECOSIM --beamspot Gauss --fileout file:reco.root --filein file:raw.root --python_filename RecoMuon_Fullsim_cfg.py --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R39F16_cff")
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1.2.1 $'),
    annotation = cms.untracked.string('step2/Phase1/R34F16 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('OtherCMS', 
        'StdException', 
        'Unknown', 
        'BadAlloc', 
        'BadExceptionType', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FileOpenError', 
        'FileReadError', 
        'FatalRootError', 
        'MismatchedInputFiles', 
        'ProductDoesNotSupportViews', 
        'ProductDoesNotSupportPtr', 
        'NotFound')
)
# Input source
process.source = cms.Source("PoolSource",
#  fileNames = cms.untracked.vstring('file:/uscms_data/d2/cheung/slhc/r34v25/muon/TenMuon_RAWSIM_0pu.root')
  fileNames = cms.untracked.vstring('/store/user/cheung/slhc_r34v25_336_10mu/slhc_r34v25_336_10mu/beecc745d5b81c616cb1a592abb88bf6/Muon_RAWSIM_1_1.root')
)

# Output definition
#process.output = cms.OutputModule("PoolOutputModule",
#    splitLevel = cms.untracked.int32(0),
#    outputCommands = process.RECOSIMEventContent.outputCommands,
#    fileName = cms.untracked.string('file:/uscms_data/d2/cheung/slhc/r34v25/muon/reco.root'),
#    dataset = cms.untracked.PSet(
#        dataTier = cms.untracked.string('GEN-SIM-RECO'),
#        filterName = cms.untracked.string('')
#    )
#)
#extendAOD = cms.untracked.vstring(
#  'drop *',
#        'keep *_source_*_*',
#        'keep *_VtxSmeared_*_*',
#        'keep SimTracks_g4SimHits_*_*',
#        'keep SimVertexs_g4SimHits_*_*',
#        'keep *_offlinePrimaryVertices_*_*',
#        'keep *_offlinePrimaryVerticesWithBS_*_*',
#        'keep recoTracks_ctfWithMaterialTracks_*_*')
##        'keep recoTracks_generalTracks_*_*')
#
#process.AODSIMEventContent.outputCommands.extend(extendAOD)
#
#process.output = cms.OutputModule("PoolOutputModule",
#         outputCommands = process.AODSIMEventContent.outputCommands,
#         fileName = cms.untracked.string('file:/uscms_data/d2/cheung/slhc/r34v25/muon/reco.root')
#)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'MC_3XY_V9A::All'

### PhaseI Geometry and modifications ###############################################

process.load("SLHCUpgradeSimulations.Geometry.PhaseI_cmsSimIdealGeometryXML_cff")
process.Timing =  cms.Service("Timing")
process.mix.playback = True
#process.mix.input.nbPileupEvents = cms.PSet(
#  averageNumber = cms.double(5.0)
#  #sigmaInel = cms.double(80.0),
#  #Lumi = cms.double(2.0)
#)
#process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo_fullph1geom")

process.load("SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_cff")
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")

process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'

### Now Validation and other user functions #########################################
process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("Validation.RecoTrack.MultiTrackValidator_cff")
#process.multiTrackValidator.label = ['generalTracks']
### if using simple (non-iterative) or old (as in 1_8_4) tracking
process.multiTrackValidator.label = ['ctfWithMaterialTracks']
#process.multiTrackValidator.label = ['cutsRecoTracks']
#process.multiTrackValidator.label_tp_effic = cms.InputTag("cutsTPEffic")
#process.multiTrackValidator.label_tp_fake = cms.InputTag("cutsTPFake")
process.multiTrackValidator.associators = ['TrackAssociatorByHits']
process.multiTrackValidator.UseAssociators = True
process.multiTrackValidator.outputFile = "validfullph1g_muon_50GeV.root"
process.multiTrackValidator.nint = cms.int32(20)
process.multiTrackValidator.nintpT = cms.int32(25)
process.multiTrackValidator.maxpT = cms.double(50.0)
process.multiTrackValidator.skipHistoFit = False

##### with John's changes ##############################
process.load("SLHCUpgradeSimulations.Geometry.oldTracking_wtriplets")
process.PixelLayerTriplets.layerList = cms.vstring('BPix1+BPix2+BPix3',
        'BPix1+BPix3+BPix4',
        'BPix2+BPix3+BPix4',
        'BPix1+BPix2+BPix4',
        'BPix1+BPix2+FPix1_pos',
        'BPix1+BPix2+FPix1_neg',
        'BPix1+FPix1_pos+FPix2_pos',
        'BPix1+FPix1_neg+FPix2_neg',
        'BPix1+FPix2_pos+FPix3_pos',
        'BPix1+FPix2_neg+FPix3_neg',
        'FPix1_pos+FPix2_pos+FPix3_pos',
        'FPix1_neg+FPix2_neg+FPix3_neg')
# restrict vertex fining in trackingtruthprod to smaller volume (note: these numbers in mm)
## needs to be in the step 1
#process.mergedtruth.volumeRadius = cms.double(100.0)
#process.mergedtruth.volumeZ = cms.double(900.0)
#process.mergedtruth.discardOutVolume = cms.bool(True)

process.cutsTPFake.tip = cms.double(10.0)
process.cutsTPFake.lip = cms.double(90.0)
#NB: tracks are already filtered by the generalTracks sequence
#for additional cuts use the cutsRecoTracks filter:
#process.load("Validation.RecoTrack.cutsRecoTracks_cfi")
#process.cutsRecoTracks.src = cms.InputTag("ctfWithMaterialTracks")
#process.cutsRecoTracks.quality = cms.string('')
#process.cutsRecoTracks.minHit = cms.int32(3)
#process.cutsRecoTracks.minHit = cms.int32(8)
#process.cutsRecoTracks.minHit = cms.int32(6)
############ end John's changes ###########################
process.ReadLocalMeasurement = cms.EDAnalyzer("StdHitNtuplizer",
   src = cms.InputTag("siPixelRecHits"),
   stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
   rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
   matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
   #trackProducer = cms.InputTag("generalTracks"),
   ### if using simple (non-iterative) or old (as in 1_8_4) tracking
   trackProducer = cms.InputTag("ctfWithMaterialTracks"),
   OutputFile = cms.string("stdgrechitfullph1g_ntuple.root"),
   ### for using track hit association
   associatePixel = cms.bool(True),
   associateStrip = cms.bool(False),
   associateRecoTracks = cms.bool(False),
   ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof',
                         'g4SimHitsTrackerHitsPixelBarrelHighTof',
                         'g4SimHitsTrackerHitsPixelEndcapLowTof',
                         'g4SimHitsTrackerHitsPixelEndcapHighTof')
)
process.anal = cms.EDAnalyzer("EventContentAnalyzer")

### back to standard commands
## extra settings
process.load('RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsSimpleGaussian_cff')
process.es_prefer_beamspot = cms.ESPrefer("BeamSpotFakeConditions","") 

process.load("RecoVertex.Configuration.RecoVertex_cff")
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ctfWithMaterialTracks")
process.offlinePrimaryVerticesWithBS.TrackLabel = cms.InputTag("ctfWithMaterialTracks")
process.generalV0Candidates.trackRecoAlgorithm = cms.InputTag('ctfWithMaterialTracks')

process.load("SLHCUpgradeSimulations.Validation.SLHCPrimaryVertexAnalyzer_cfi")
process.slhcSimpleVertexAnalysis.recoTrackProducer = cms.untracked.string('ctfWithMaterialTracks')

# Path and EndPath definitions
#process.reconstruction_step = cms.Path(process.reconstruction)
process.mix_step = cms.Path(process.mix)
process.reconstruction_step = cms.Path(process.trackerlocalreco*process.offlineBeamSpot+process.oldTracking_wtriplets)
process.debug_step = cms.Path(process.anal)
process.validation_step = cms.Path(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.user_step = cms.Path(process.vertexreco*process.slhcSimpleVertexAnalysis*process.ReadLocalMeasurement)
process.user_step = cms.Path(process.vertexreco*process.slhcSimpleVertexAnalysis)
process.endjob_step = cms.Path(process.endOfProcess)
#process.out_step = cms.EndPath(process.output)

# Schedule definition
#process.schedule = cms.Schedule(process.mix_step,process.reconstruction_step,process.debug_step,process.validation_step,process.user_step,process.endjob_step,process.out_step)
#process.schedule = cms.Schedule(process.mix_step,process.reconstruction_step,process.validation_step,process.user_step,process.endjob_step,process.out_step)
process.schedule = cms.Schedule(process.mix_step,process.reconstruction_step,process.validation_step,process.user_step,process.endjob_step)
#process.schedule = cms.Schedule(process.mix_step,process.reconstruction_step,process.validation_step,process.endjob_step)
