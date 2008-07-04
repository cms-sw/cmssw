import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#
simEcalTriggerPrimitiveDigis = cms.EDProducer("EcalTrigPrimProducer",
    BarrelOnly = cms.bool(False),
    InstanceEB = cms.string('ebDigis'),
    InstanceEE = cms.string('eeDigis'),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10xsxsxd
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Label = cms.string('simEcalDigis')
)


