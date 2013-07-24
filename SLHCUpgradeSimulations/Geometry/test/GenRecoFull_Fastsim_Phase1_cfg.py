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
#process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')
process.load('SLHCUpgradeSimulations.Geometry.mixLowLumPU_FastSim14TeV_cff')
#process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('FastSimulation.Configuration.Geometries_cff')
process.load('SLHCUpgradeSimulations.Geometry.Phase1_R34F16_cmsSimIdealGeometryXML_cff')
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

process.load('SLHCUpgradeSimulations.Geometry.Digi_Phase1_R34F16_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R34F16_cff')
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_phase1_cff")

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
process.famosPileUp.PileUpSimulator.averageNumber = 0.0
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
    version = cms.untracked.string('$Revision: 1.6 $'),
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
   OutputFile = cms.string("stdgrechitP1_ntuple.root"),
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

process.p1 = cms.Path(process.famosWithTrackerAndCaloHits)
process.p2 = cms.Path(process.trDigi*process.trackingParticles)
process.reconstruction_step     = cms.Path(process.trackerlocalreco*
                                           process.offlineBeamSpot+
                                           process.recopixelvertexing*
                                           process.ckftracks_wodEdX*process.trackExtrapolator*
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
process.p9 = cms.Path(process.ReadLocalMeasurement)

process.endjob_step             = cms.Path(process.endOfProcess)
process.out_step                = cms.EndPath(process.output)

#process.schedule = cms.Schedule(process.generation_step,process.othergeneration_step,process.genfiltersummary_step,process.p1,process.p2,process.reconstruction_step,process.p9,process.endjob_step,process.out_step)
process.schedule = cms.Schedule(process.generation_step,process.othergeneration_step,process.genfiltersummary_step,process.p1,process.p2,process.reconstruction_step,process.endjob_step,process.out_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

