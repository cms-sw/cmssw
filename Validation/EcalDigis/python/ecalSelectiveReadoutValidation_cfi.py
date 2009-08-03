import FWCore.ParameterSet.Config as cms

ecalSelectiveReadoutValidation = cms.EDFilter("EcalSelectiveReadoutValidation",
    #Input collection names:
    EbDigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    EeDigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    EbUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
    EeUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis"),
    EbSrFlagCollection = cms.InputTag("simEcalDigis","ebSrFlags"),
    EeSrFlagCollection = cms.InputTag("simEcalDigis","eeSrFlags"),
    EbSrFlagFromTTCollection = cms.InputTag("simEcalDigis","ebSrFlagsFromTT"),
    EeSrFlagFromTTCollection = cms.InputTag("simEcalDigis","eeSrFlagsFromTT"),

    EbSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
    EeSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
    TrigPrimCollection = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    EbRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EeRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    FEDRawCollection = cms.InputTag("source"),

    #Versbose mode switch:
    verbose = cms.untracked.bool(False),

    #Name of the output histrogram root file:
    outputFile = cms.untracked.string('srvalid_hists.root'),

    #Switch to enable local amplitude reconstruction from digis instead
    #of RecHit's:
    LocalReco = cms.bool(True),

    #Weights used for the local reconstruction of the signal amplitude:
    weights = cms.vdouble(-0.295252, -0.295252, -0.295252, -0.286034, 0.240376,
                          0.402839, 0.322126, 0.172504, 0.0339461, 0.0),

    # Index of time sample (starting from 1) the first DCC weights is implied.
    # Used for histogram of DCC filter output.
    ecalDccZs1stSample = cms.int32(2),

    #DCC ZS FIR weights: weights are rounded in such way that in Hw
    #representation (weigth*1024 rounded to nearest integer) the sum is null.
    #Used for the histogram of DCC filter output.
    dccWeights = cms.vdouble(-0.374, -0.374, -0.3629, 0.2721,
                             0.4681, 0.3707),

    #ZS threshold used to validate ZS application
    #Threshold in ADC count. Resolution of 1/4th ADC count.
    zsThrADCCount = cms.double(9./4.),

    # Index of time sample (starting from 1) the first DCC weights is implied.
    # Used for histogram of DCC filter output.
    ecalDccZs1stSample = cms.int32(2),

    #DCC ZS FIR weights: weights are rounded in such way that in Hw
    #representation (weigth*1024 rounded to nearest integer) the sum is null.
    #Used for the histogram of DCC filter output.
    dccWeights = cms.vdouble(-0.374, -0.374, -0.3629, 0.2721,
                             0.4681, 0.3707),
 
    #Switch to express TP in GeV for the histograms:
    tpInGeV = cms.bool(True),

    #ROOT/DQM directory where to store the histograms
    histDir = cms.string('EcalDigisV/EcalDigiTask'),

    #Switch to fill histograms with event rate instead of event count.
    #Applies only to some histograms.
    useEventRate = cms.bool(True),

    #List of histograms to produce. Run the module once with LogInfo enabled
    #in order to get the list of available histograms. If the list contains
    #the keyword "all", every available histogram is produced.
    histograms = cms.untracked.vstring('all'),


    #List of FEDs to exclude in comparison of data with emulation
    excludedFeds = cms.vint32(),

    #File to log SRP algorithm errors (differences between SRF from
    #data file and SRF from SRP emulation). If empty logging
    #is disabled:
    srpAlgoErrorLogFile = cms.untracked.string(''),

    #File to log SRP decission application errors. If empty logging
    #is disabled:
    srApplicationErrorLogFile = cms.untracked.string(''),
)



