import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#
simEcalEBTriggerPrimitiveDigis = cms.EDProducer("EcalEBTrigPrimProducer",
    BarrelOnly = cms.bool(True),
    barrelEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis"),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
    UseRecHits = cms.bool(False),
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    nOfSamples = cms.int32(1)
)


