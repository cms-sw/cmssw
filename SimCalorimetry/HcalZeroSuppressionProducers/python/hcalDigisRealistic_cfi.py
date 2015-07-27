# to use the ZS thresholds from config file, 
# set useConfigZSvalues = cms.int32(1)
# to generate Unsuppressed digis, 
# also need to set useConfigZSvalues = cms.int32(1) and -inf. (-999) levels
# to use the channel-by-channel ZS values from DB, 
# set useConfigZSvalues = cms.int32(0) - default

import FWCore.ParameterSet.Config as cms

simHcalDigis = cms.EDProducer("HcalRealisticZS",
    digiLabel = cms.string("simHcalUnsuppressedDigis"),
    markAndPass = cms.bool(False),
    HBlevel = cms.int32(8),
    HElevel = cms.int32(9),
    HOlevel = cms.int32(24),
    HFlevel = cms.int32(-9999),
    HBregion = cms.vint32(3,6),      
    HEregion = cms.vint32(3,6),
    HOregion = cms.vint32(1,8),
    HFregion = cms.vint32(2,3),      
    useConfigZSvalues = cms.int32(0)
)



