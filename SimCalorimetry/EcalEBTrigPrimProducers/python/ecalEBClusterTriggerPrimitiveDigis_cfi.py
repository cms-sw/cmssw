import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#
simEcalEBClusterTriggerPrimitiveDigis = cms.EDProducer("EcalEBCluTrigPrimProducer",
    BarrelOnly = cms.bool(True),
    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis"),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    nOfSamples = cms.int32(1),
    etaSize = cms.int32(2), # to build the 3x3 or 3x5 or whatever. the int is always size-1. So fra a 3x3, one needs to input 2,2 
    phiSize = cms.int32(2),
    hitNoiseCut = cms.double(0.175),
    etCutOnSeed = cms.double(0.4375) # 2.5x0.175 see Sasha slides


)


