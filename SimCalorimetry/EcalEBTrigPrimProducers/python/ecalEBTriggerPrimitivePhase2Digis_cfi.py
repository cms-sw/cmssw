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
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
phase2_ecalTP_devel.toModify( simEcalEBTriggerPrimitivePhase2Digis)
(phase2_ecalTP_devel & premix_stage2).toModify(simEcalEBTriggerPrimitivePhase2Digis, barrelEcalDigis = 'mixData')
