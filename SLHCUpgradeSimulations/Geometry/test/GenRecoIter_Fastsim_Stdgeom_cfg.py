# Auto generated configuration file
# using: 
# Revision: 1.303.2.3 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SLHCUpgradeSimulations/Configuration/python/FourMuPt_1_50_cfi.py -s GEN,FASTSIM,HLT:GRun --pileup=NoPileUp --geometry DB -n 10 --conditions auto:mc --eventcontent FEVTDEBUG --datatier GEN-SIM-DIGI-RECO --beamspot Gauss --no_exec --python_filename FASTSIM_4muons_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('FASTSIMWDIGI')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff')
#process.load('SLHCUpgradeSimulations.Geometry.mixLowLumPU_FastSim14TeV_cff')
process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')
process.load('FastSimulation.Configuration.Geometries_MC_cff')
#process.load('FastSimulation.Configuration.Geometries_cff')
process.load('FastSimulation.Configuration.FamosSequences_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
#process.load('Configuration.StandardSequences.Generator_cff')
#process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedParameters_cfi')
#process.load('FastSimulation.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.RandomNumberGeneratorService.simSiStripDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))
process.RandomNumberGeneratorService.simSiPixelDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))

process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load("SLHCUpgradeSimulations.Geometry.fakeConditions_stdgeom_cff")
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_stdgeom_cff")

process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(True)  #FG set True to use errors from templates
process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(True)  #FG set True to load the last version of the templates
process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
process.PixelCPEGenericESProducer.DoCosmics = False

## CPE for other steps
process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.initialStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.lowPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.pixelPairStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.detachedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.mixedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.pixelLessStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.tobTecStepTracks.TTRHBuilder = cms.string('WithTrackAngle')

# Need these lines to stop some errors about missing siStripDigis collections.
# should add them to fakeConditions_Phase1_cff
process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.lowPtTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.lowPtTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.lowPtTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.lowPtTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.lowPtTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.lowPtTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.pixelPairStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.pixelPairStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.pixelPairStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.pixelPairStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.pixelPairStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.pixelPairStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.detachedTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.detachedTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.detachedTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.mixedTripletStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.mixedTripletStepMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.mixedTripletStepMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.mixedTripletStepMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.mixedTripletStepMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.mixedTripletStepMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.pixelLessStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.tobTecStepMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

## for fastsim we need these ################################
process.TrackerGeometricDetESModule.fromDDD=cms.bool(True)
process.TrackerDigiGeometryESModule.fromDDD=cms.bool(True)
process.simSiPixelDigis.ROUList =  ['famosSimHitsTrackerHits']
process.simSiStripDigis.ROUList =  ['famosSimHitsTrackerHits']
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.mergedtruth.simHitCollections.tracker = ['famosSimHitsTrackerHits']
process.mergedtruth.simHitCollections.pixel = []
process.mergedtruth.simHitCollections.muon = []
process.mergedtruth.simHitLabel = 'famosSimHits'
## make occupancies more similar to full simulation
process.famosSimHits.ParticleFilter.etaMax = 3.0
process.famosSimHits.ParticleFilter.pTMin = 0.05
process.famosSimHits.TrackerSimHits.pTmin = 0.05
process.famosSimHits.TrackerSimHits.firstLoop = False
#############################################################
process.Timing =  cms.Service("Timing")

# If you want to turn on/off pile-up, default is no pileup
process.famosPileUp.PileUpSimulator.averageNumber = 0.0
### if doing inefficiency at <PU>=50
#process.simSiPixelDigis.AddPixelInefficiency = 20

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('SLHCUpgradeSimulations/Configuration/python/FourMuPt_1_50_cfi.py nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
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
#I'm only interested in the validation stuff for this job
process.output.outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*')

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'DESIGN42_V17::All'

process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.GaussVtxSmearingParameters.type = cms.string("Gaussian")
process.famosSimHits.VertexGenerator = process.GaussVtxSmearingParameters
process.famosPileUp.VertexGenerator = process.GaussVtxSmearingParameters

####################
#process.load("Configuration.Generator.PythiaUESettings_cfi")
#process.generator = cms.EDFilter("Pythia6GeneratorFilter",
#    pythiaHepMCVerbosity = cms.untracked.bool(False),
#    maxEventsToPrint = cms.untracked.int32(0),
#    pythiaPylistVerbosity = cms.untracked.int32(0),
#    filterEfficiency = cms.untracked.double(1.0),
#    comEnergy = cms.double(14000.0),
#    PythiaParameters = cms.PSet(
#        process.pythiaUESettingsBlock,
#        processParameters = cms.vstring('MSEL      = 0     ! User defined processes', 
#            'MSUB(81)  = 1     ! qqbar to QQbar', 
#            'MSUB(82)  = 1     ! gg to QQbar', 
#            'MSTP(7)   = 6     ! flavour = top', 
#            'PMAS(6,1) = 175.  ! top quark mass'),
#        # This is a vector of ParameterSet names to be read, in this order
#        parameterSets = cms.vstring('pythiaUESettings', 
#            'processParameters')
#    )
#)
################
process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(200.0),
        MinPt = cms.double(0.9),
        PartID = cms.vint32(-13, -13),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359)
    ),  
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('Four mu pt 1 to 200'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
##########################################################
process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")

process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.trackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']
process.load('SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi')
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

process.load('Configuration.StandardSequences.Validation_cff')
#
# for fastsim we need the following
process.trackValidator.stableOnlyTP = True
process.trackValidator.histoProducerAlgoBlock.generalTpSelector.stableOnly = True
process.trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta.stableOnly = True
process.trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi.stableOnly = True
process.trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.stableOnly = True
process.trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR.stableOnly = True
process.trackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ.stableOnly = True

import PhysicsTools.RecoAlgos.recoTrackSelector_cfi

process.cutsRecoTracksHpwbtagc = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
process.cutsRecoTracksHpwbtagc.quality=cms.vstring("highPurity")
process.cutsRecoTracksHpwbtagc.minHit=cms.int32(8)
process.cutsRecoTracksHpwbtagc.ptMin = cms.double(1.0)

process.trackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                           cms.InputTag("cutsRecoTracksHp"),
                                           cms.InputTag("cutsRecoTracksHpwbtagc"),
                                           cms.InputTag("cutsRecoTracksInitialStepHp"),
                                           cms.InputTag("cutsRecoTracksLowPtTripletStepHp")
                                           )
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
                                 process.quickTrackAssociatorByHits*
                                 process.trackValidator)

########################################
### produce an ntuple with hits for analysis
process.ReadLocalMeasurement = cms.EDAnalyzer("StdHitNtuplizer",
   src = cms.InputTag("siPixelRecHits"),
   stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
   rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
   matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
   simtrackHits = cms.InputTag("famosSimHits"),
   trackProducer = cms.InputTag("generalTracks"),
   ### if using simple (non-iterative) or old (as in 1_8_4) tracking
   #trackProducer = cms.InputTag("ctfWithMaterialTracks"),
   OutputFile = cms.string("stdgrechitStd1_ntuple.root"),
   ### for using track hit association
   associatePixel = cms.bool(True),
   associateStrip = cms.bool(False),
   associateRecoTracks = cms.bool(False),
   ROUList = cms.vstring('famosSimHitsTrackerHits')
)
process.ReadFastsimHits = cms.EDAnalyzer("FastsimHitNtuplizer",
   HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits"),
   VerbosityLevel = cms.untracked.int32(1),
   OutputFile = cms.string("fsgrechitP1_ntuple.root")
)

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

########################################
# Path and EndPath definitions
# Famos with tracks
process.p0 = cms.Path(process.generator)
process.p1 = cms.Path(process.famosWithTrackerHits)
process.p2 = cms.Path(process.trDigi*process.trackingParticles)
process.reconstruction_step     = cms.Path(process.trackerlocalreco*
                                                process.offlineBeamSpot+
                                                process.recopixelvertexing*process.ckftracks_wodEdX)
process.p7 = cms.Path(process.anal)
process.p8 = cms.Path(process.cutsTPEffic*process.cutsTPFake*process.slhcTracksValidation)
process.p9 = cms.Path(process.ReadLocalMeasurement)

process.endjob_step             = cms.Path(process.endOfProcess)
process.out_step                = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.reconstruction_step,process.p8,process.p9,process.endjob_step,process.out_step)
#process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.reconstruction_step,process.p8,process.endjob_step,process.out_step)

