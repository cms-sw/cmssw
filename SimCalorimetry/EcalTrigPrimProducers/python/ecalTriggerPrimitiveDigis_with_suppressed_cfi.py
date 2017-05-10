import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#
simEcalTriggerPrimitiveDigis = cms.EDProducer("EcalTrigPrimProducer",
    BarrelOnly = cms.bool(False),
    EBlabel = cms.InputTag("simEcalDigis","ebDigis"),
    EElabel = cms.InputTag("simEcalDigis","eeDigis"),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Famos = cms.bool(False),
    binOfMaximum = cms.int32(6) ## optional from release 200 on, from 1-10xsxsxd
)
