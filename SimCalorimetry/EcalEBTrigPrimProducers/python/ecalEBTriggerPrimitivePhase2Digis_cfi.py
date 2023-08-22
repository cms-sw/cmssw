import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#

simEcalEBTriggerPrimitivePhase2Digis = cms.EDProducer("EcalEBTrigPrimPhase2Producer",
    BarrelOnly = cms.bool(True),
    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis"),
    binOfMaximum = cms.int32(8), ## optional from release 200 on, from 1-10
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False)
)


