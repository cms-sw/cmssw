import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripGainSimESProducer_cfi import *

simSiStripDigis = cms.EDFilter("SiStripDigitizer",
                               #---SiLinearChargeDivider
                               DeltaProductionCut      = cms.double(0.120425),
                               APVpeakmode             = cms.bool(False), # also SiStripDigitizerAlgorithm
                               LandauFluctuations      = cms.bool(True),
                               chargeDivisionsPerStrip = cms.int32(10),
                               CosmicDelayShift        = cms.untracked.double(0.0), # also SiStripDigitizerAlgorithm
                               #---SiHitDigitizer
                               DepletionVoltage        = cms.double(170.0),
                               AppliedVoltage          = cms.double(300.0),
                               ChargeMobility          = cms.double(310.0),
                               Temperature             = cms.double(273.0),
                               GevPerElectron          = cms.double(3.61e-09),
                               ChargeDistributionRMS   = cms.double(6.5e-10),
                               noDiffusion             = cms.bool(False),
                               #---SiTrivialInduceChargeOnStrips
                               #TIB
                               CouplingConstantDecIB1  = cms.vdouble(0.7746, 0.0962, 0.0165),
                               CouplingConstantDecIB2  = cms.vdouble(0.8300, 0.0756, 0.0094),
                               CouplingConstantPeakIB1 = cms.vdouble(0.9006, 0.0497),
                               CouplingConstantPeakIB2 = cms.vdouble(0.9344, 0.0328),
                               #TOB
                               CouplingConstantDecOB1  = cms.vdouble(0.6872, 0.1222, 0.0342),
                               CouplingConstantDecOB2  = cms.vdouble(0.7468, 0.1015, 0.0251),
                               CouplingConstantPeakOB1 = cms.vdouble(0.8542, 0.0729),
                               CouplingConstantPeakOB2 = cms.vdouble(0.8720, 0.0640),
                               #TID
                               CouplingConstantDecW1a  = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW2a  = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW3a  = cms.vdouble(0.76, 0.12),
                               CouplingConstantPeakW1a = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW2a = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW3a = cms.vdouble(0.94, 0.03),
                               #TEC
                               CouplingConstantDecW1b  = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW2b  = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW3b  = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW4   = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW5   = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW6   = cms.vdouble(0.76, 0.12),
                               CouplingConstantDecW7   = cms.vdouble(0.76, 0.12),
                               CouplingConstantPeakW1b = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW2b = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW3b = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW4  = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW5  = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW6  = cms.vdouble(0.94, 0.03),
                               CouplingConstantPeakW7  = cms.vdouble(0.94, 0.03),
                               #-----SiStripDigitizer
                               DigiModeList = cms.PSet(SCDigi = cms.string('ScopeMode'),
                                                       ZSDigi = cms.string('ZeroSuppressed'),
                                                       PRDigi = cms.string('ProcessedRaw'),
                                                       VRDigi = cms.string('VirginRaw')),
                               ROUList = cms.vstring("g4SimHitsTrackerHitsTIBLowTof","g4SimHitsTrackerHitsTIBHighTof",
                                                     "g4SimHitsTrackerHitsTIDLowTof","g4SimHitsTrackerHitsTIDHighTof",
                                                     "g4SimHitsTrackerHitsTOBLowTof","g4SimHitsTrackerHitsTOBHighTof",
                                                     "g4SimHitsTrackerHitsTECLowTof","g4SimHitsTrackerHitsTECHighTof"),
                               GeometryType               = cms.string('idealForDigi'),
                               TrackerConfigurationFromDB = cms.bool(False),
                               ZeroSuppression            = cms.bool(True),
                               LorentzAngle               = cms.string(''),
                               Gain                       = cms.string(''),
                               #-----SiStripDigitizerAlgorithm
                               NoiseSigmaThreshold        = cms.double(2.0),
                               electronPerAdc             = cms.double(217.0), #obsolete, check that it can be safely removed
                               electronPerAdcDec          = cms.double(217.0), #https://twiki.cern.ch/twiki/bin/view/CMS/TRKTuningDecoMC
                               electronPerAdcPeak         = cms.double(262.0), #https://twiki.cern.ch/twiki/bin/view/CMS/TRKTuningPeakMC
                               FedAlgorithm               = cms.int32(4),
                               Noise                      = cms.bool(True), ## NOTE : turning Noise ON/OFF will make a big change
                               TOFCutForDeconvolution     = cms.double(50.0),
                               TOFCutForPeak              = cms.double(100.0),
			       Inefficiency               = cms.double(0.0)
                              )
