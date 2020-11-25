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

    # Switch for reading SRP settings from condition database
    configFromCondDB = cms.bool(True),

    # Switch to turn off SRP altogether using special DB payload
    UseFullReadout = cms.bool(False),

    # ES label?
    # NZSLabel = cms.ESInputTag(' '),

    # Label name of input ECAL trigger primitive collection
    trigPrimProducer = cms.string('simEcalTriggerPrimitiveDigis'),

    # Instance name of ECAL trigger primitive collection
    trigPrimCollection = cms.string(''),

    #switch to run w/o trigger primitive. For debug use only

    trigPrimBypass = cms.bool(False),

    # Mode selection for "Trig bypass" mode
    # 0: TT thresholds applied on sum of crystal Et's
    # 1: TT thresholds applies on compressed Et from Trigger primitive
    # @ee trigPrimByPass_ switch
    trigPrimBypassMode = cms.int32(0),
                              
    #for debug mode only:
    trigPrimBypassLTH = cms.double(1.0),

    #for debug mode only:
    trigPrimBypassHTH = cms.double(1.0),

    #for debug mode only
    trigPrimBypassWithPeakFinder = cms.bool(True),
                              
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
    defaultTtf = cms.int32(4)
)

# Turn off SR in Ecal for premixing stage1
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(simEcalDigis, UseFullReadout = True)

_simEcalDigisPh2 = simEcalDigis.clone(
    trigPrimBypass = True,
)
