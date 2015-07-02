# Auto generated configuration file
# using: 
# Revision: 1.172.2.5 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s RECO -n 100 --conditions DESIGN_36_V10::All --datatier GEN-SIM-RECO --eventcontent RECOSIM --beamspot Gauss --fileout file:reco.root --filein file:raw.root --python_filename RecoMuon_Fullsim_cfg.py --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff")
process.load('SLHCUpgradeSimulations.Geometry.Longbarrel_cmsSimIdealGeometryXML_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('SLHCUpgradeSimulations.Geometry.Digi_Longbarrel_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.22 $'),
    annotation = cms.untracked.string('step2 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)

)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       'file:FourMuPt_1_200_cfi_GEN_SIM.root'
    )
)
# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:valid_reco.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    )
)
#I'm only interested in the validation stuff
process.output.outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*')

#process.output = cms.OutputModule("PoolOutputModule",
#         outputCommands = process.AODSIMEventContent.outputCommands,
#         fileName = cms.untracked.string(
#		'file:/uscms_data/d2/brownson/slhc/quadMuon_RECO.root')
#)


# Additional output definition

# Other statements
#process.GlobalTag.globaltag = 'MC_42_V10::All'
process.GlobalTag.globaltag = 'DESIGN42_V17::All'

### PhaseI Geometry and modifications ###############################################
process.Timing =  cms.Service("Timing")
## no playback when doing digis
#process.mix.playback = True
#process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo_fullph1geom")

### if pileup we need to set the number
#process.mix.input.nbPileupEvents = cms.PSet(
#  averageNumber = cms.double(50.0)
#)
### if doing inefficiency at <PU>=50
#process.simSiPixelDigis.AddPixelInefficiency = 20
## also for strips TIB inefficiency if we want
## TIB1,2 inefficiency at 20%
#process.simSiStripDigis.Inefficiency = 20
## TIB1,2 inefficiency at 50%
#process.simSiStripDigis.Inefficiency = 30
## TIB1,2 inefficiency at 99% (i.e. dead)
#process.simSiStripDigis.Inefficiency = 40

process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_Longbarrel_cff')
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_Longbarrel_cff")
process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_longbarrel_cff")

process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
process.PixelCPEGenericESProducer.SmallPitch = False
process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
process.PixelCPEGenericESProducer.DoCosmics = False

## CPE for other steps
process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.initialStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.highPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.lowPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.pixelPairStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.detachedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.mixedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.pixelLessStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.tobTecStepTracks.TTRHBuilder = cms.string('WithTrackAngle')

## so strip simhits are not asked for
process.mergedtruth.simHitCollections.pixel = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof',
                         'g4SimHitsTrackerHitsPixelBarrelHighTof',
                         'g4SimHitsTrackerHitsPixelEndcapLowTof',
                         'g4SimHitsTrackerHitsPixelEndcapHighTof')
process.mergedtruth.simHitCollections.tracker = []
process.mergedtruth.simHitCollections.muon = []
# Need these lines to stop some errors about missing siStripDigis collections.
# should add them to fakeConditions_Phase1_cff
process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
#process.lowPtTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
#process.lowPtTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
#process.lowPtTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
#process.lowPtTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
#process.lowPtTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
#process.lowPtTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
#process.pixelPairStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
#process.pixelPairStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
#process.pixelPairStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
#process.pixelPairStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
#process.pixelPairStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
#process.pixelPairStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.detachedTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.detachedTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
#process.mixedTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
#process.mixedTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
#process.mixedTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
#process.mixedTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
#process.mixedTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
#process.mixedTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.pixelLessStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.tobTecStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

### Now Validation and other user functions #########################################
process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")

process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load('SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi')
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

process.load('Configuration.StandardSequences.Validation_cff')
### look look at OOTB generalTracks and high purity collections
### for high purity also look at 8 hit and pT>1 requirements
### some definitions in Validation/RecoTrack/python/TrackValidation_cff.py

import PhysicsTools.RecoAlgos.recoTrackSelector_cfi

process.cutsRecoTracksHpwbtagc = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
process.cutsRecoTracksHpwbtagc.quality=cms.vstring("highPurity")
process.cutsRecoTracksHpwbtagc.minHit=cms.int32(8)
process.cutsRecoTracksHpwbtagc.ptMin = cms.double(1.0)

process.trackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                           cms.InputTag("cutsRecoTracksHp"),
                                           cms.InputTag("cutsRecoTracksHpwbtagc"),
                                           cms.InputTag("cutsRecoTracksInitialStepHp"),
                                           cms.InputTag("cutsRecoTracksLowPtTripletStepHp"),
                                           cms.InputTag("cutsRecoTracksPixelPairStepHp"),
                                           cms.InputTag("cutsRecoTracksDetachedTripletHp"),
                                           cms.InputTag("cutsRecoTracksMixedTripletHp")
                                           )
#process.trackValidator.associators = ['trackAssociatorByHits']
process.trackValidator.associators = ['quickTrackAssociatorByHits']
process.trackValidator.UseAssociators = True
## options to match with 363 histos for comparison
process.trackValidator.histoProducerAlgoBlock.nintEta = cms.int32(20)
process.trackValidator.histoProducerAlgoBlock.nintPt = cms.int32(100)
process.trackValidator.histoProducerAlgoBlock.maxPt = cms.double(200.0)
process.trackValidator.histoProducerAlgoBlock.useLogPt = cms.untracked.bool(True)
process.trackValidator.histoProducerAlgoBlock.minDxy = cms.double(-3.0)
process.trackValidator.histoProducerAlgoBlock.maxDxy = cms.double(3.0)
process.trackValidator.histoProducerAlgoBlock.nintDxy = cms.int32(100)
process.trackValidator.histoProducerAlgoBlock.minDz = cms.double(-10.0)
process.trackValidator.histoProducerAlgoBlock.maxDz = cms.double(10.0)
process.trackValidator.histoProducerAlgoBlock.nintDz = cms.int32(100)
process.trackValidator.histoProducerAlgoBlock.maxVertpos = cms.double(5.0)
process.trackValidator.histoProducerAlgoBlock.nintVertpos = cms.int32(100)
process.trackValidator.histoProducerAlgoBlock.minZpos = cms.double(-10.0)
process.trackValidator.histoProducerAlgoBlock.maxZpos = cms.double(10.0)
process.trackValidator.histoProducerAlgoBlock.nintZpos = cms.int32(100)
process.trackValidator.histoProducerAlgoBlock.phiRes_rangeMin = cms.double(-0.003)
process.trackValidator.histoProducerAlgoBlock.phiRes_rangeMax = cms.double(0.003)
process.trackValidator.histoProducerAlgoBlock.phiRes_nbin = cms.int32(100)
process.trackValidator.histoProducerAlgoBlock.cotThetaRes_rangeMin = cms.double(-0.01)
process.trackValidator.histoProducerAlgoBlock.cotThetaRes_rangeMax = cms.double(+0.01)
process.trackValidator.histoProducerAlgoBlock.cotThetaRes_nbin = cms.int32(120)
process.trackValidator.histoProducerAlgoBlock.dxyRes_rangeMin = cms.double(-0.01)
process.trackValidator.histoProducerAlgoBlock.dxyRes_rangeMax = cms.double(0.01)
process.trackValidator.histoProducerAlgoBlock.dxyRes_nbin = cms.int32(100)
process.trackValidator.tipTP = cms.double(3.5)
process.trackValidator.ptMinTP = cms.double(0.9)

process.slhcTracksValidation = cms.Sequence(process.cutsRecoTracksHp*
                                 process.cutsRecoTracksHpwbtagc*
                                 process.cutsRecoTracksInitialStepHp*
                                 process.cutsRecoTracksLowPtTripletStepHp*
                                 process.cutsRecoTracksPixelPairStepHp*
                                 process.cutsRecoTracksDetachedTripletHp*
                                 process.cutsRecoTracksMixedTripletHp*
                                 process.quickTrackAssociatorByHits*
                                 process.trackValidator)

############ end John's changes ###########################
process.ReadLocalMeasurement = cms.EDAnalyzer("StdHitNtuplizer",
   src = cms.InputTag("siPixelRecHits"),
   stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
   rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
   matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
   ### if using simple (non-iterative) or old (as in 1_8_4) tracking
   trackProducer = cms.InputTag("generalTracks"),
   OutputFile = cms.string("stdgrechitfullLBg_ntuple.root"),
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

## for seed info
#process.load("tracking.TrackRecoMonitoring.seedmultiplicitymonitor_cfi")
#process.seedmultiplicitymonitor.seedCollections = cms.VPSet(cms.PSet(src=cms.InputTag("initialStepSeeds")),
# cms.PSet(src=cms.InputTag("highPtTripletStepSeeds")),
# cms.PSet(src=cms.InputTag("lowPtTripletStepSeeds")),
# cms.PSet(src=cms.InputTag("pixelPairStepSeeds"),
#          maxValue=cms.untracked.double(200000),nBins=cms.untracked.uint32(2000)),
# cms.PSet(src=cms.InputTag("mixedTripletStepSeeds"),
#          maxValue=cms.untracked.double(200000),nBins=cms.untracked.uint32(2000)),
# cms.PSet(src=cms.InputTag("newCombinedSeeds"),
#          maxValue=cms.untracked.double(200000),nBins=cms.untracked.uint32(2000))
#)
#process.TFileService= cms.Service("TFileService",
#                                  fileName= cms.string("histograms_seedmult.root")
#                                  )

## need this at the end as the validation config redefines random seed with just mix
#process.load("IOMC.RandomEngine.IOMC_cff")

### back to standard job commands ##################################################

# Path and EndPath definitions
process.mix_step 		= cms.Path(process.mix)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)

process.reconstruction_step 	= cms.Path(process.trackerlocalreco*
						process.offlineBeamSpot+
                                                process.recopixelvertexing*process.ckftracks_wodEdX)
process.debug_step 		= cms.Path(process.anal)
process.validation_step 	= cms.Path(process.cutsTPEffic*
						process.cutsTPFake*
						process.slhcTracksValidation)
#process.user_step 		= cms.Path(process.seedmultiplicitymonitor*process.ReadLocalMeasurement)
process.user_step             = cms.Path(process.ReadLocalMeasurement)
process.endjob_step 		= cms.Path(process.endOfProcess)
process.out_step 		= cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.reconstruction_step,process.validation_step,process.user_step,process.endjob_step,process.out_step)
#process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.reconstruction_step,process.validation_step,process.endjob_step,process.out_step)

