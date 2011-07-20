import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")



# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff')
#process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')
process.load('SLHCUpgradeSimulations.Geometry.mixLowLumPU_FastSim14TeV_cff')
#process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('FastSimulation.Configuration.Geometries_cff')
process.load('SLHCUpgradeSimulations.Geometry.Phase1_R39F16_cmsSimIdealGeometryXML_cff')
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

process.load('SLHCUpgradeSimulations.Geometry.Digi_Phase1_R39F16_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R39F16_cff')
process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_phase1_cff")

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
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.0 $'),
    annotation = cms.untracked.string('SLHCUpgradeSimulations/Configuration/python/FourMuPt_1_50_cfi.py nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Other statements
process.GlobalTag.globaltag = 'DESIGN42_V11::All'

process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.GaussVtxSmearingParameters.type = cms.string("Gaussian")
process.famosSimHits.VertexGenerator = process.GaussVtxSmearingParameters
process.famosPileUp.VertexGenerator = process.GaussVtxSmearingParameters

####################
from Configuration.Generator.PythiaUEZ2Settings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
                             pythiaHepMCVerbosity = cms.untracked.bool(False),
                             maxEventsToPrint = cms.untracked.int32(0),
                             pythiaPylistVerbosity = cms.untracked.int32(1),
                             filterEfficiency = cms.untracked.double(1.0),
                             crossSection = cms.untracked.double(0.9738),
                             comEnergy = cms.double(7000.0),
                             ExternalDecays = cms.PSet(
            Tauola = cms.untracked.PSet(
                TauolaPolar,
                            TauolaDefaultInputCards
                        ),
                    parameterSets = cms.vstring('Tauola')
                ),
                             UseExternalGenerators = cms.untracked.bool(True),
                             PythiaParameters = cms.PSet(
            pythiaUESettingsBlock,
                    processParameters = cms.vstring('MSEL=0            !User defined processes',
                                                                                            'MSUB(1)=1         !Incl Z0/gamma* production',
                                                                                            'MSTP(43)=3        !Both Z0 and gamma*',
                                                                                            'MDME(174,1)=0     !Z decay into d dbar',
                                                                                            'MDME(175,1)=0     !Z decay into u ubar',
                                                                                            'MDME(176,1)=0     !Z decay into s sbar',
                                                                                            'MDME(177,1)=0     !Z decay into c cbar',
                                                                                            'MDME(178,1)=0     !Z decay into b bbar',
                                                                                            'MDME(179,1)=0     !Z decay into t tbar',
                                                                                            'MDME(182,1)=1     !Z decay into e- e+',
                                                                                            'MDME(183,1)=0     !Z decay into nu_e nu_ebar',
                                                                                            'MDME(184,1)=0     !Z decay into mu- mu+',
                                                                                            'MDME(185,1)=0     !Z decay into nu_mu nu_mubar',
                                                                                            'MDME(186,1)=0     !Z decay into tau- tau+',
                                                                                            'MDME(187,1)=0     !Z decay into nu_tau nu_taubar',
                                                                                            'CKIN(1)=200.       !Minimum sqrt(s_hat) value (=Z mass)'),
                    # This is a vector of ParameterSet names to be read, in this order
                    parameterSets = cms.vstring('pythiaUESettings',
                                                            'processParameters')
                )
                         )
process.load('Configuration.StandardSequences.Validation_cff')

process.anal = cms.EDAnalyzer("EventContentAnalyzer")

process.load('FastSimulation.CaloRecHitsProducer.CaloRecHits_cff')
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
# Calo Towers
from RecoJets.Configuration.CaloTowersRec_cff import *

# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = False

#Load Scales
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")

process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

process.ecalRecHit.doDigis = True
process.hbhereco.doDigis = True
process.horeco.doDigis = True
process.hfreco.doDigis = True


process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysisCalibrated_cfi")

process.genParticles = cms.EDProducer("GenParticleProducer",
                                      saveBarCodes = cms.untracked.bool(True),
                                      src = cms.InputTag("generator"),
                                      abortOnUnknownPDGCode = cms.untracked.bool(False)
                                      )


process.p1 = cms.Path(process.generator+
                      process.genParticles+
                      process.famosWithTrackerAndCaloHits+
                      process.simEcalTriggerPrimitiveDigis+
                      process.simHcalTriggerPrimitiveDigis+
                      process.SLHCCaloTrigger+
                      process.mcSequence+
                      process.analysisSequenceCalibrated

)



process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histograms.root")
)

