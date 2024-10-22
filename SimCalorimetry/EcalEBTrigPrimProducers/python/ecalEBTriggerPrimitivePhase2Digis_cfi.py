import FWCore.ParameterSet.Config as cms

#
# attention: default is changed to work on unsuppressed digis!! ##############
#

simEcalEBTriggerPrimitivePhase2Digis = cms.EDProducer("EcalEBTrigPrimPhase2Producer",
    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis"),
    binOfMaximum = cms.int32(6), 
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False)
)


from Configuration.Eras.Modifier_phase2_ecalTP_devel_cff import phase2_ecalTP_devel
phase2_ecalTP_devel.toModify( simEcalEBTriggerPrimitivePhase2Digis)
