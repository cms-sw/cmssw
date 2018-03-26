import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#

process.simEcalEBClusterTriggerPrimitiveDigis = cms.EDProducer("EcalEBCluTrigPrimProducer",
                                                               BarrelOnly = cms.bool(True),
                                                               barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis"),
                                                               #    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
                                                               TcpOutput = cms.bool(False),
                                                               Debug = cms.bool(False),
                                                               Famos = cms.bool(False),
                                                               nOfSamples = cms.int32(1),
                                                               binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
                                                               etaSize = cms.int32(2), # to build the 3x3 or 3x5 or whatever. the int is always size-1. So for a 3x3, one needs to input 2,2 
                                                               phiSize = cms.int32(2)
)

