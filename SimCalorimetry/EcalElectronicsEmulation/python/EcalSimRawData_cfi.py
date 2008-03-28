import FWCore.ParameterSet.Config as cms

#define an EcalSimRawData module, named 'ecalSimRawData'
#simulation of raw data:
ecalSimRawData = cms.EDFilter("EcalSimRawData",
    dccNum = cms.untracked.int32(-1),
    writeMode = cms.string('ascii'),
    EEdigiCollection = cms.string(''),
    EBdigiCollection = cms.string(''),
    tcpDigiCollection = cms.string('formatTCP'),
    srProducer = cms.string('ecalDigis'),
    outputBaseName = cms.untracked.string('ecal'),
    tpVerbose = cms.untracked.bool(False),
    trigPrimProducer = cms.string('ecalTriggerPrimitiveDigis'),
    xtalVerbose = cms.untracked.bool(False),
    srp2dccData = cms.untracked.bool(True),
    EESrFlagCollection = cms.string('eeSrFlags'),
    fe2tccData = cms.untracked.bool(True),
    unsuppressedDigiProducer = cms.string('ecalUnsuppressedDigis'),
    fe2dccData = cms.untracked.bool(True),
    EBSrFlagCollection = cms.string('ebSrFlags'),
    tccNum = cms.untracked.int32(-1),
    tcc2dccData = cms.untracked.bool(True),
    tccInDefaultVal = cms.untracked.int32(65535),
    trigPrimDigiCollection = cms.string('')
)


