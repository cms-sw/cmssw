import FWCore.ParameterSet.Config as cms

ecalSelectiveReadoutValidation = cms.EDFilter("EcalSelectiveReadoutValidation",
    #Input collection names:
    EbDigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EeDigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    EbUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
    EeUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
    EbSrFlagCollection = cms.InputTag("simEcalDigis","ebSrFlags"),
    EeSrFlagCollection = cms.InputTag("simEcalDigis","eeSrFlags"),
    EbSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
    EeSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
    TrigPrimCollection = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    EbRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EeRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    FEDRawCollection = cms.InputTag("source"),

    #Versbose mode switch:
    verbose = cms.untracked.bool(True),

    #Name of the output histrogram root file:
    outputFile = cms.untracked.string('EcalSelectiveReadoutValidationHistos.root'),

    #Switch to enable locale amplitude reconstruction from digis instead 
    #of RecHit's:
    LocalReco = cms.bool(True),

    #Weights used for the local reconstruction of the signal amplitude:
    weights = cms.vdouble(-0.295252, -0.295252, -0.295252, -0.286034, 0.240376, 
        0.402839, 0.322126, 0.172504, 0.0339461, 0.0),

    #Switch to express TP in GeV for the histograms:
    tpInGeV = cms.bool(True),

    #ROOT/DQM directory where to store the histograms
    histDir = cms.string('EcalDigisV/EcalDigiTask'),

    #List of histograms to produce. Run the module once with LogInfo enabled
    #in order to get the list of available histograms. If the list contains
    #the keyword "all", every available histogram is produced.
    histograms = cms.untracked.vstring('all')
)



