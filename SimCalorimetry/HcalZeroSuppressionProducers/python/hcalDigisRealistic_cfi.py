# to use the ZS thresholds from config file, 
# set useConfigZSvalues = cms.int32(1)
# to generate Unsuppressed digis, 
# also need to set useConfigZSvalues = cms.int32(1) and -inf. (-999) levels
# to use the channel-by-channel ZS values from DB, 
# set useConfigZSvalues = cms.int32(0) - default

import FWCore.ParameterSet.Config as cms

simHcalDigis = cms.EDProducer("HcalRealisticZS",
    digiLabel = cms.string("simHcalUnsuppressedDigis"),
    useInstanceLabels = cms.bool(True),
    markAndPass = cms.bool(False),
    HBlevel = cms.int32(8),
    HElevel = cms.int32(9),
    HOlevel = cms.int32(24),
    HFlevel = cms.int32(-9999),
#--- region: (3,6) for 2TS means (3+4),(4+5),(5+6) 
#--- region: (3,6) for 1TS means (3),(4),(5) i.e. < 6 in C++ "for" cycle.  
    HBregion = cms.vint32(3,6),      
    HEregion = cms.vint32(3,6),
    HOregion = cms.vint32(1,8),
    HFregion = cms.vint32(2,3),      
    useConfigZSvalues = cms.int32(0),
    use1ts = cms.bool(False)              # True for >= Run3
)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toModify( simHcalDigis,
                             HFregion = [1,2]
)

from Configuration.Eras.Modifier_run2_HB_2018_cff import run2_HB_2018
run2_HB_2018.toModify( simHcalDigis,
                             HBregion = [2,5]
)

from Configuration.Eras.Modifier_run2_HE_2018_cff import run2_HE_2018
run2_HE_2018.toModify( simHcalDigis,
                             HEregion = [2,5]
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( simHcalDigis, 
                             use1ts = True,
                             HBregion = [5,6],  # SOI in 10TS
                             HEregion = [5,6]   # SOI in 10TS
) 


# Switch off HCAL ZS in digi for premixing stage1
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(simHcalDigis,
    markAndPass = True,
    HBlevel = -999,
    HElevel = -999,
    HOlevel = -999,
    HFlevel = -999,
    useConfigZSvalues = 1
)
