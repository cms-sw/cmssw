import FWCore.ParameterSet.Config as cms

ecalDigis = cms.EDProducer("EcalSelectiveReadoutProducer",
    # ADC to GeV conversion factor used in ZS filter for EB
    ebDccAdcToGeV = cms.double(0.035),
    # Instance name of input EB digi collections
    EEdigiCollection = cms.string(''),
    # Instance name of input EB digi collections
    EBdigiCollection = cms.string(''),
    # ADC to GeV conversion factor used in ZS filter for EE
    eeDccAdcToGeV = cms.double(0.06),
    #logical flag to write out SrFlags
    writeSrFlags = cms.untracked.bool(True),
    # Instance name of output EB SR flags collection
    EBSrFlagCollection = cms.string('ebSrFlags'),
    #for debug mode only:
    trigPrimBypassLTH = cms.double(1.0),
    # Neighbour eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
    deltaEta = cms.int32(1),
    # Neighbouring eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
    deltaPhi = cms.int32(1),
    # Label name of input ECAL trigger primitive collection
    trigPrimProducer = cms.string('ecalTriggerPrimitiveDigis'),
    # ZS energy threshold in GeV to apply to low interest channels of endcap
    srpEndcapLowInterestChannelZS = cms.double(0.3),
    # Instance name of ECAL trigger primitive collection
    trigPrimCollection = cms.string(''),
    # ZS energy threshold in GeV to apply to high interest channels of endcap
    srpEndcapHighInterestChannelZS = cms.double(-1000000000.0),
    #for debug mode only
    trigPrimBypassWithPeakFinder = cms.bool(True),
    # Index of time sample (staring from 1) the first DCC weights is implied
    ecalDccZs1stSample = cms.int32(2),
    #DCC ZS FIR weights: weights are rounded in such way that in Hw
    #representation (weigth*1024 rounded to nearest integer) the sum is null:
    dccNormalizedWeights = cms.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 
        0.3707),
    # Instance name of output EE digis collection
    EESRPdigiCollection = cms.string('eeDigis'),
    #number of events whose TT and SR flags must be dumped (for debug purpose):
    dumpFlags = cms.untracked.int32(0),
    # ZS energy threshold in GeV to apply to low interest channels of barrel
    srpBarrelLowInterestChannelZS = cms.double(0.1),
    # Instance name of output EB digis collection
    EBSRPdigiCollection = cms.string('ebDigis'),
    # Label of input EB and EE digi collections
    digiProducer = cms.string('ecalUnsuppressedDigis'),
    # Instance name of output EE SR flags collection
    EESrFlagCollection = cms.string('eeSrFlags'),
    #switch to run w/o trigger primitive. For debug use only
    trigPrimBypass = cms.bool(False),
    # ZS energy threshold in GeV to apply to high interest channels of barrel
    srpBarrelHighInterestChannelZS = cms.double(-1000000000.0),
    #for debug mode only:
    trigPrimBypassHTH = cms.double(1.0)
)


