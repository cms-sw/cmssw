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
process.load("SLHCUpgradeSimulations.Geometry.Longbarrel_cmsSimIdealGeometryXML_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('SLHCUpgradeSimulations.Geometry.Digi_Longbarrel_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.16 $'),
    annotation = cms.untracked.string('step2 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
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
    #outputCommands = cms.untracked.vstring('keep *','drop *_mix_*_*'),
    fileName = cms.untracked.string('file:reco.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    )
)
#I'm only interested in the validation stuff
#process.output.outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*')

#process.output = cms.OutputModule("PoolOutputModule",
#         outputCommands = process.AODSIMEventContent.outputCommands,
#         fileName = cms.untracked.string(
#		'file:/uscms_data/d2/brownson/slhc/quadMuon_RECO.root')
#)


# Additional output definition

# Other statements
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

process.muons.TrackerKinkFinderParameters.TrackerRecHitBuilder = cms.string('WithTrackAngle')
# The SeedMergerPSet should be added to the following file for Phase 1
# RecoTracker/SpecialSeedGenerators/python/CombinatorialSeedGeneratorForCosmicsRegionalReconstruction_cfi.py
# but pixel layers are not used here for cosmic TODO: need Maria and Jan to do appropriate thing here
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import PixelSeedMergerQuadruplets
process.regionalCosmicTrackerSeeds.SeedMergerPSet = cms.PSet(
	mergeTriplets = cms.bool(False),
	ttrhBuilderLabel = cms.string( "PixelTTRHBuilderWithoutAngle" ),
	addRemainingTriplets = cms.bool(False),
	layerList = PixelSeedMergerQuadruplets
	)
process.regionalCosmicTracks.TTRHBuilder = cms.string('WithTrackAngle')

process.regionalCosmicTrackerSeedingLayers.layerList = cms.vstring(
	'BPix16+BPix15','BPix16+BPix14','BPix16+BPix13',
	'BPix15+BPix14','BPix15+BPix13',
	'BPix14+BPix13'
	)
process.regionalCosmicTrackerSeedingLayers.BPix = cms.PSet(
	TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
	)

#################################

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

## need this at the end as the validation config redefines random seed with just mix
#process.load("IOMC.RandomEngine.IOMC_cff")

### back to standard job commands ##################################################
process.DigiToRaw.remove(process.castorRawData)

process.DigiToRaw.remove(process.siPixelRawData)
process.RawToDigi.remove(process.siPixelDigis)

## removing large memory usage module if we don't need it
process.pdigi.remove(process.mergedtruth)

process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)

# Path and EndPath definitions
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)

process.reconstruction_step 	= cms.Path(process.reconstruction)
process.mix_step 		= cms.Path(process.mix)
process.debug_step 		= cms.Path(process.anal)
process.user_step 		= cms.Path(process.ReadLocalMeasurement)
process.endjob_step 		= cms.Path(process.endOfProcess)
process.out_step 		= cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.out_step)

