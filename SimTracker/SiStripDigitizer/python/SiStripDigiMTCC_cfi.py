import FWCore.ParameterSet.Config as cms

siStripDigis = cms.EDFilter("SiStripDigitizer",
    DeltaProductionCut = cms.double(0.120425),
    Temperature = cms.double(263.0),
    #-------------------------------------
    #-----SiHitDigitizer
    DepletionVoltage = cms.double(140.0),
    SigmaShapePeak = cms.double(52.17),
    #-----SiStripDigitizerAlgorithm 
    electronPerAdc = cms.double(250.0),
    SimProducer = cms.string('SimDigitizer'),
    VerbosityLevel = cms.untracked.int32(0),
    #---------------------------------------
    #SiTrivialInduceChargeOnStrips
    CouplingCostantDeco = cms.vdouble(0.76, 0.12),
    AppliedVoltage = cms.double(150.0),
    ZeroSuppression = cms.bool(True),
    Noise = cms.bool(True), ## NOTE : turning Noise ON/OFF will make a big change

    noDiffusion = cms.bool(False),
    #--------------------------------------
    #SiLinearChargeDivider
    LandauFluctuations = cms.bool(True),
    #--------------------------------
    #---ZeroSuppression
    FedAlgorithm = cms.int32(4),
    AdcFullScale = cms.int32(255),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof'),
    GevPerElectron = cms.double(3.61e-09),
    chargeDivisionsPerStrip = cms.int32(10),
    ChargeMobility = cms.double(480.0),
    APVpeakmode = cms.bool(True), ## also in SiLinearChargeDivider

    SigmaShapeDeco = cms.double(12.06),
    NoiseSigmaThreshold = cms.double(2.0),
    ChargeDistributionRMS = cms.double(6.5e-10),
    CouplingCostantPeak = cms.vdouble(0.76, 0.12),
    CosmicDelayShift = cms.untracked.double(10.0)
)


