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
process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')
#process.load('SLHCUpgradeSimulations.Geometry.mixLowLumPU_FastSim14TeV_cff')
#process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('FastSimulation.Configuration.Geometries_cff')
process.load('SLHCUpgradeSimulations.Geometry.Longbarrel_cmsSimIdealGeometryXML_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('FastSimulation.Configuration.FamosSequences_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedParameters_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.RandomNumberGeneratorService.simSiStripDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))
process.RandomNumberGeneratorService.simSiPixelDigis = cms.PSet(
      initialSeed = cms.untracked.uint32(1234567),
      engineName = cms.untracked.string('HepJamesRandom'))

process.load('SLHCUpgradeSimulations.Geometry.Digi_Longbarrel_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

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
process.newPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.secPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.thPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.preFilterZeroStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.preFilterStepOneTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.secWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.thWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.fourthWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.fifthWithMaterialTracks.TTRHBuilder = cms.string('WithTrackAngle')

# Need these lines to stop some errors about missing siStripDigis collections.
# should add them to fakeConditions_Phase1_cff
process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.newMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.newMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.newMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.newMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.newMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.newMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.secMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.secMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.secMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.secMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.secMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.secMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.thMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.thMeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.thMeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.thMeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.thMeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.thMeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)
process.fourthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.fifthMeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()

process.muons.TrackerKinkFinderParameters.TrackerRecHitBuilder = cms.string('WithTrackAngle')
process.regionalCosmicTrackerSeeds.SeedMergerPSet = cms.PSet(
        mergeTriplets = cms.bool(False),
        ttrhBuilderLabel = cms.string( "PixelTTRHBuilderWithoutAngle" ),
        addRemainingTriplets = cms.bool(False),
        layerListName = cms.string( "PixelSeedMergerQuadruplets" )
        )
process.regionalCosmicTracks.TTRHBuilder = cms.string('WithTrackAngle')

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
#process.famosPileUp.PileUpSimulator.averageNumber = 50.00
### if doing inefficiency at <PU>=50
#process.simSiPixelDigis.AddPixelInefficiency = 20

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
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
    fileName = cms.untracked.string('file:reco.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    )
)

# Other statements
process.GlobalTag.globaltag = 'DESIGN42_V17::All'

process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.GaussVtxSmearingParameters.type = cms.string("Gaussian")
process.famosSimHits.VertexGenerator = process.GaussVtxSmearingParameters
process.famosPileUp.VertexGenerator = process.GaussVtxSmearingParameters

####################
process.load("Configuration.Generator.PythiaUESettings_cfi")
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    comEnergy = cms.double(14000.0),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL      = 0     ! User defined processes', 
            'MSUB(81)  = 1     ! qqbar to QQbar', 
            'MSUB(82)  = 1     ! gg to QQbar', 
            'MSTP(7)   = 6     ! flavour = top', 
            'PMAS(6,1) = 175.  ! top quark mass'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    ),
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
             UseTauolaPolarization = cms.bool(True),
             InputCards = cms.PSet
             (
                pjak1 = cms.int32(0),
                pjak2 = cms.int32(0),
                mdtau = cms.int32(0)
             )
        ),
        parameterSets = cms.vstring('Tauola')
    )
)
################
#process.generator = cms.EDProducer("FlatRandomPtGunProducer",
#    PGunParameters = cms.PSet(
#        MaxPt = cms.double(50.0),
#        MinPt = cms.double(0.9),
#        PartID = cms.vint32(-13, -13),
#        MaxEta = cms.double(2.5),
#        MaxPhi = cms.double(3.14159265359),
#        MinEta = cms.double(-2.5),
#        MinPhi = cms.double(-3.14159265359)
#    ),
#    Verbosity = cms.untracked.int32(0),
#    psethack = cms.string('Four mu pt 1 to 50'),
#    AddAntiParticle = cms.bool(True),
#    firstRun = cms.untracked.uint32(1)
#)

##########################################################
## for MultiTrackValidator
process.load("Validation.RecoTrack.cutsTPEffic_cfi")
process.load("Validation.RecoTrack.cutsTPFake_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.TrackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']
## Mark's alternate faster associator
#process.load('SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi')
#process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

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
                                           cms.InputTag("cutsRecoTracksZeroHp"),
                                           cms.InputTag("cutsRecoTracksFirstHp")
                                           )
#process.trackValidator.associators = cms.vstring('quickTrackAssociatorByHits')
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
                                 process.cutsRecoTracksZeroHp*
                                 process.cutsRecoTracksFirstHp*
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
   OutputFile = cms.string("stdgrechitLB_ntuple.root"),
   ### for using track hit association
   associatePixel = cms.bool(True),
   associateStrip = cms.bool(False),
   associateRecoTracks = cms.bool(False),
   ROUList = cms.vstring('famosSimHitsTrackerHits')
)
process.ReadFastsimHits = cms.EDAnalyzer("FastsimHitNtuplizer",
   HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSRecHits"),
   VerbosityLevel = cms.untracked.int32(1),
   OutputFile = cms.string("fsgrechitLB_ntuple.root")
)

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

########################################
## other subdetectors besides tracking uses fastsim

process.load('FastSimulation.CaloRecHitsProducer.CaloRecHits_cff')
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
# Calo Towers
from RecoJets.Configuration.CaloTowersRec_cff import *

process.load('RecoTracker.Configuration.RecoTracker_cff')
# Calo RecHits producer (with no HCAL miscalibration by default)

# Muon RecHit sequence
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
rpcRecHits.rpcDigiLabel = 'simMuonRPCDigis'
dt1DRecHits.dtDigiLabel = 'simMuonDTDigis'
dt1DCosmicRecHits.dtDigiLabel = 'simMuonDTDigis'

# Muon reconstruction sequence
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
KFSmootherForMuonTrackLoader.Propagator = 'SmartPropagatorAny'
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *
from FastSimulation.Configuration.globalMuons_cff import *
globalMuons.GLBTrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TransformerOutPropagator = cms.string('SmartPropagatorAny')
globalMuons.GLBTrajBuilderParameters.MatcherOutPropagator = cms.string('SmartPropagator')

from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *
GlobalMuonRefitter.TrackerRecHitBuilder = 'WithoutRefit'
GlobalMuonRefitter.Propagator = 'SmartPropagatorAny'
GlobalTrajectoryBuilderCommon.TrackerRecHitBuilder = 'WithoutRefit'
tevMuons.RefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'
tevMuons.RefitterParameters.Propagator =  'SmartPropagatorAny'
KFSmootherForRefitInsideOut.Propagator = 'SmartPropagatorAny'
KFSmootherForRefitOutsideIn.Propagator = 'SmartPropagator'
KFFitterForRefitInsideOut.Propagator = 'SmartPropagatorAny'
KFFitterForRefitOutsideIn.Propagator = 'SmartPropagatorAny'


#from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
#from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
#from RecoEgamma.EgammaPhotonProducers.conversionSequence_cff import *
#from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
#from RecoEgamma.EgammaPhotonProducers.allConversionSequence_cff import *
#allConversions.src = 'gsfGeneralConversionTrackMerger'
########################################

# Famos with tracks
process.p0 = cms.Path(process.generator)
process.generation_step = cms.Path(process.pgen_genonly)
process.othergeneration_step = cms.Path(process.GeneInfo+process.genJetMET)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)

#process.p1 = cms.Path(process.famosWithTrackerHits)
process.p1 = cms.Path(process.famosWithTrackerAndCaloHits)
process.p2 = cms.Path(process.trDigi*process.trackingParticles)
#process.p3 = cms.Path(process.trackerlocalreco)
#process.p6 = cms.Path(process.oldTracking_wtriplets)
process.reconstruction_step     = cms.Path(process.trackerlocalreco*
                                           process.offlineBeamSpot+
                                           process.recopixelvertexing*
                                           process.ckftracks_wodEdXandSteps2345*process.trackExtrapolator*
                                           process.particleFlowCluster*
                                           process.ecalClusters*
                                          process.caloTowersRec*
                                           process.vertexreco*
###                                           process.egammaGlobalReco*
                                           process.electronGsfTracking*process.conversionTrackSequence*process.conversionTrackSequenceNoEcalSeeded*
                                           process.allConversionSequence*
                                           process.pfTrackingGlobalReco*
                                           process.jetGlobalReco*
                                           process.famosMuonSequence*
                                           process.famosMuonIdAndIsolationSequence*
###                                           process.highlevelreco
                                           process.egammaHighLevelRecoPrePF*
                                           process.particleFlowReco*
                                           process.egammaHighLevelRecoPostPF*
                                           process.jetHighLevelReco*
                                           process.tautagging*
###                                           process.metrecoPlusHCALNoise*
                                           process.btagging*
                                           process.recoPFMET*
                                           process.PFTau*
                                           process.regionalCosmicTracksSeq*
###                                           process.muoncosmichighlevelreco*
                                           process.reducedRecHits
)

process.p7 = cms.Path(process.anal)
process.p8 = cms.Path(process.cutsTPEffic*process.cutsTPFake*process.slhcTracksValidation)
process.p9 = cms.Path(process.ReadLocalMeasurement)

process.endjob_step             = cms.Path(process.endOfProcess)
process.out_step                = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.generation_step,process.othergeneration_step,process.genfiltersummary_step,process.p1,process.p2,process.reconstruction_step,process.p8,process.p9,process.endjob_step,process.out_step)
#process.schedule = cms.Schedule(process.generation_step,process.othergeneration_step,process.genfiltersummary_step,process.p1,process.p2,process.reconstruction_step,process.endjob_step,process.out_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

