import FWCore.ParameterSet.Config as cms

simEcalDigis = cms.EDProducer("EcalSelectiveReadoutProducer",
    # Label of input EB and EE digi collections
    digiProducer = cms.string('simEcalUnsuppressedDigis'),

    # Instance name of input EB digi collections
    EBdigiCollection = cms.string(''),

    # Instance name of input EB digi collections
    EEdigiCollection = cms.string(''),

    # Instance name of output EB SR flags collection
    EBSrFlagCollection = cms.string('ebSrFlags'),

    # Instance name of output EE SR flags collection
    EESrFlagCollection = cms.string('eeSrFlags'),

    # Instance name of output EB digis collection
    EBSRPdigiCollection = cms.string('ebDigis'),

    # Instance name of output EE digis collection
    EESRPdigiCollection = cms.string('eeDigis'),

    # Label name of input ECAL trigger primitive collection
    trigPrimProducer = cms.string('simEcalTriggerPrimitiveDigis'),

    # Instance name of ECAL trigger primitive collection
    trigPrimCollection = cms.string(''),

    # Neighbour eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
    deltaEta = cms.int32(1),

    # Neighbouring eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
    deltaPhi = cms.int32(1),

    # Index of time sample (staring from 1) the first DCC weights is implied
    ecalDccZs1stSample = cms.int32(3),

    # ADC to GeV conversion factor used in ZS filter for EB
    ebDccAdcToGeV = cms.double(0.035),

    # ADC to GeV conversion factor used in ZS filter for EE
    eeDccAdcToGeV = cms.double(0.06),

    #DCC ZS FIR weights.
    #d-efault value set of DCC firmware used in CRUZET and CRAFT
    dccNormalizedWeights = cms.vdouble(-1.1865, 0.0195, 0.2900, 0.3477, 0.3008,
                                        0.2266),

    # Switch to use a symetric zero suppression (cut on absolute value). For
    # studies only, for time being it is not supported by the hardware.
    symetricZS = cms.bool(False),

    # ZS energy threshold in GeV to apply to low interest channels of barrel
    srpBarrelLowInterestChannelZS = cms.double(3*.035),

    # ZS energy threshold in GeV to apply to low interest channels of endcap
    srpEndcapLowInterestChannelZS = cms.double(3*0.06),

    # ZS energy threshold in GeV to apply to high interest channels of barrel
    srpBarrelHighInterestChannelZS = cms.double(-1.e9),

    # ZS energy threshold in GeV to apply to high interest channels of endcap
    srpEndcapHighInterestChannelZS = cms.double(-1.e9),

    #switch to run w/o trigger primitive. For debug use only
    trigPrimBypass = cms.bool(False),
                              
    #for debug mode only:
    trigPrimBypassLTH = cms.double(1.0),

    #for debug mode only:
    trigPrimBypassHTH = cms.double(1.0),

    #for debug mode only
    trigPrimBypassWithPeakFinder = cms.bool(True),

    # Mode selection for "Trig bypass" mode
    # 0: TT thresholds applied on sum of crystal Et's
    # 1: TT thresholds applies on compressed Et from Trigger primitive
    # @ee trigPrimByPass_ switch
    trigPrimBypassMode = cms.int32(0),
                              
    #number of events whose TT and SR flags must be dumped (for debug purpose):
    dumpFlags = cms.untracked.int32(0),
                              
    #logical flag to write out SrFlags
    writeSrFlags = cms.untracked.bool(True),

    #switch to apply selective readout decision on the digis and produce
    #the "suppressed" digis
    produceDigis = cms.untracked.bool(True),

    #Trigger Tower Flag to use when a flag is not found from the input
    #Trigger Primitive collection. Must be one of the following values:
    # 0: low interest, 1: mid interest, 3: high interest
    # 4: forced low interest, 5: forced mid interest, 7: forced high interest
    defaultTtf_ = cms.int32(4),

    # SR->action flag map
    actions = cms.vint32(1, 3, 3, 3, 5, 7, 7, 7)
)



