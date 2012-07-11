import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(200)
        )

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")


EventType = "BJets_Pt_50_120_7TeV" #"ZEE" #"TTbar" #"PhotonJet_Pt_120_170" 

# Generate event
#process.load("SLHCUpgradeSimulations.L1CaloTrigger.QQH1352T_cfi")
process.load("Configuration.Generator."+EventType+"_cfi")


# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_GRun_cff")

# Only event accepted by L1 + HLT are reconstructed
#process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)
process.HLTEndSequence = cms.Sequence()

from FastSimulation.Configuration.HLT_GRun_cff import *


process.HLTSchedule = cms.Schedule( *(
	HLTriggerFirstPath,
	HLT_L1Jet6U,
	HLT_L1Jet10U,
	HLT_Jet15U,
	HLT_Jet30U,
	HLT_Jet50U,
	HLT_Jet70U,
	HLT_Jet100U,
	HLT_DiJetAve15U,
	HLT_DiJetAve30U,
	HLT_DiJetAve50U,
	HLT_DiJetAve70U,
	HLT_DoubleJet15U_ForwardBackward,
	HLT_DoubleJet25U_ForwardBackward,
	HLT_ExclDiJet30U,
	HLT_QuadJet15U,
	HLT_QuadJet20U,
	HLT_QuadJet25U,
	HLT_L1ETT100,
	HLT_EcalOnly_SumEt160,
	HLT_L1MET20,
	HLT_MET45,
	HLT_MET65,
	HLT_MET100,
	HLT_HT100U,
	HLT_HT120U,
	HLT_HT140U,
	HLT_L1SingleEG2,
	HLT_L1SingleEG5,
	HLT_L1SingleEG8,
	HLT_L1DoubleEG5,
	HLT_Ele10_SW_L1R,
	HLT_Ele12_SW_TightEleId_L1R,
	HLT_Ele12_SW_TightEleIdIsol_L1R,
	HLT_Ele12_SW_TightEleIdIsol_NoDEtaInEE_L1R,
	HLT_Ele17_SW_L1R,
	HLT_Ele17_SW_CaloEleId_L1R,
	HLT_Ele17_SW_LooseEleId_L1R,
	HLT_Ele17_SW_EleId_L1R,
	HLT_Ele22_SW_CaloEleId_L1R,
	HLT_Ele40_SW_L1R,
	HLT_DoubleEle10_SW_L1R,
	HLT_Photon10_Cleaned_L1R,
	HLT_Photon15_Cleaned_L1R,
	HLT_Photon20_NoHE_L1R,
	HLT_Photon20_Cleaned_L1R,
	HLT_Photon30_Cleaned_L1R,
	HLT_Photon50_NoHE_L1R,
	HLT_Photon50_NoHE_Cleaned_L1R,
	HLT_DoublePhoton5_CEP_L1R,
	HLT_DoublePhoton5_L1R,
	HLT_DoublePhoton10_L1R,
	HLT_DoublePhoton15_L1R,
	HLT_DoublePhoton17_L1R,
	HLT_SingleIsoTau20_Trk5_MET20,
	HLT_SingleIsoTau20_Trk15_MET20,
	HLT_SingleIsoTau30_Trk5_MET20,
	HLT_SingleIsoTau30_Trk5_L120or30,
	HLT_DoubleIsoTau15_OneLeg_Trk5,
	HLT_DoubleIsoTau15_Trk5,
#	HLT_Activity_CSC,
#	HLT_L1MuOpen,
#	HLT_L1MuOpen_DT,
#	HLT_L1Mu,
#	HLT_L1Mu20,
#	HLT_L2Mu0_NoVertex,
#	HLT_L2Mu0,
#	HLT_L2Mu3,
#	HLT_L2Mu9,
#	HLT_L2Mu25,
#	HLT_Mu3,
#	HLT_Mu5,
#	HLT_Mu7,
#	HLT_Mu9,
#	HLT_Mu11,
#	HLT_IsoMu9,
#	HLT_Mu20_NoVertex,
#	HLT_L1DoubleMuOpen,
#	HLT_L2DoubleMu0,
#	HLT_DoubleMu0,
#	HLT_DoubleMu3,
#	HLT_Mu0_L1MuOpen,
#	HLT_Mu3_L1MuOpen,
#	HLT_Mu5_L1MuOpen,
#	HLT_Mu0_L2Mu0,
#	HLT_Mu5_L2Mu0,
#	HLT_BTagMu_Jet10U,
#	HLT_BTagMu_Jet20U,
#	HLT_StoppedHSCP,
#	HLT_L2Mu5_Photon9_L1R,
#	HLT_Mu5_Photon9_Cleaned_L1R,
#	HLT_ZeroBias,
#	HLT_ZeroBiasPixel_SingleTrack,
#	HLT_MinBiasPixel_SingleTrack,
#	HLT_MultiVertex6,
#	HLT_MultiVertex8_L1ETT60,
#	HLT_L1_BptxXOR_BscMinBiasOR,
#	HLT_L1Tech_BSC_minBias_OR,
#	HLT_L1Tech_BSC_minBias,
#	HLT_L1Tech_BSC_halo,
#	HLT_L1Tech_BSC_halo_forPhysicsBackground,
#	HLT_L1Tech_BSC_HighMultiplicity,
#	HLT_L1Tech_RPC_TTU_RBst1_collisions,
#	HLT_L1Tech_HCAL_HF,
#	HLT_TrackerCosmics,
#	HLT_RPCBarrelCosmics,
#	HLT_PixelTracks_Multiplicity70,
#	HLT_PixelTracks_Multiplicity85,
#	HLT_PixelTracks_Multiplicity100,
#	HLT_GlobalRunHPDNoise,
#	HLT_TechTrigHCALNoise,
#	HLT_L1_BPTX,
#	HLT_L1_BPTX_MinusOnly,
#	HLT_L1_BPTX_PlusOnly,
	HLT_LogMonitor,
	DQM_TriggerResults,
	HLTriggerFinalPath
))


# Schedule the HLT paths
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)

# Simulation sequence
#process.simulation = cms.Sequence(process.ProductionFilterSequence*process.simulationWithFamos)
process.source = cms.Source("EmptySource")
process.simulation = cms.Sequence(process.generator*process.simulationWithFamos)

# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 25

# Get frontier conditions   - not applied in the HCAL, see below
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

# Apply ECAL and HCAL miscalibration 
process.ecalRecHit.doMiscalib = True
process.hbhereco.doMiscalib = True
process.horeco.doMiscalib = True
process.hfreco.doMiscalib = True

# Apply Tracker misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.misalignedDTGeometry.applyAlignment = True
process.misalignedCSCGeometry.applyAlignment = True

# Attention ! for the HCAL IDEAL==STARTUP
# process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_0.0.xml"

#process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysisCalibrated_cfi")

# Famos with everything !
#Load Scales
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")
#process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")
process.ecalRecHit.doDigis = True
process.hbhereco.doDigis = True
process.horeco.doDigis = True
process.hfreco.doDigis = True

#process.p1 = cms.Path(process.generator+
#                      process.famosWithEverything
#                      process.simEcalTriggerPrimitiveDigis+
#                      process.simHcalTriggerPrimitiveDigis+
                    #  process.SLHCCaloTrigger+
                    #  process.mcSequence+
                    #  process.analysisSequenceCalibrated
#)

#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything)
#process.schedule.append(process.p1)
# END OF SLHC STUFF

# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *_*_*_*',
                                                                     'keep *_genParticles_*_*',
                                                                     'keep *_simEcalTriggerPrimitiveDigis_*_*',
                                                                     'keep *_simHcalTriggerPrimitiveDigis_*_*', 
                                                                     'keep L1Calo*_*_*_*',
                                                                     'keep L1Gct*_*_*_*',
                                                                     'keep *_*l1extraParticles*_*_*'
#                                                                     'keep *_*SLHCL1ExtraParticles*_*_*',
#                                                                     'keep recoGen*_*_*_*',

                                                                     ),
                     
                              fileName = cms.untracked.string(EventType+'.root')
                              )
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.outpath)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('histograms.root')
                                   )


# Keep the logging output to a nice level #
# process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.categories.append('L1GtTrigReport')
# process.MessageLogger.categories.append('HLTrigReport')
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )



#CALO TRIGGER CONFIGURATION OVERRIDE

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")
process.RCTConfigProducers.eMaxForHoECut = cms.double(60.0)
process.RCTConfigProducers.hOeCut = cms.double(0.05)
process.RCTConfigProducers.eGammaECalScaleFactors = cms.vdouble(1.0, 1.01, 1.02, 1.02, 1.02,
                                                      1.06, 1.04, 1.04, 1.05, 1.09,
                                                      1.1, 1.1, 1.15, 1.2, 1.27,
                                                      1.29, 1.32, 1.52, 1.52, 1.48,
                                                      1.4, 1.32, 1.26, 1.21, 1.17,
                                                      1.15, 1.15, 1.15)
process.RCTConfigProducers.eMinForHoECut = cms.double(3.0)
process.RCTConfigProducers.hActivityCut = cms.double(4.0)
process.RCTConfigProducers.eActivityCut = cms.double(4.0)
process.RCTConfigProducers.jetMETHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0)
process.RCTConfigProducers.eicIsolationThreshold = cms.uint32(7)
process.RCTConfigProducers.etMETLSB = cms.double(0.5)
process.RCTConfigProducers.jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0)
process.RCTConfigProducers.eMinForFGCut = cms.double(100.0)
process.RCTConfigProducers.eGammaLSB = cms.double(0.5)

process.L1GctConfigProducers = cms.ESProducer("L1GctConfigProducers",
                                          JetFinderCentralJetSeed = cms.double(0.5),
                                          JetFinderForwardJetSeed = cms.double(0.5),
                                          TauIsoEtThreshold = cms.double(2.0),
                                          HtJetEtThreshold = cms.double(10.0),
                                          MHtJetEtThreshold = cms.double(10.0),
                                          RctRegionEtLSB = cms.double(0.25),
                                          GctHtLSB = cms.double(0.25),
                                          # The CalibrationStyle should be "none", "PowerSeries", or "ORCAStyle
                                          CalibrationStyle = cms.string('None'),
                                          ConvertEtValuesToEnergy = cms.bool(False)
)                                      
