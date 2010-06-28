import FWCore.ParameterSet.Config as cms

hcalTTPRecord = cms.EDProducer("HcalTTPTriggerRecord",
    ttpDigiCollection = cms.InputTag( 'hcalTTPDigis' ),
    ttpBits           = cms.vuint32( 8,9,10 ), 
    ttpBitNames       = cms.vstring( 'L1Tech_HCAL_HF_MM_or_PP_or_PM.v0',
                                     'L1Tech_HCAL_HF_coincidence_PM.v1',
                                     'L1Tech_HCAL_HF_MMP_or_MPP.v0' )
)
