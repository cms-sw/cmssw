# The following comments couldn't be translated into the new config version:

# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR

# to use the ZS thresholds from config file, set useConfigZSvalues = cms.int32(1)
# to generate Unsuppressed digis, also need to set useConfigZSvalues = cms.int32(1)
# to use the channel-by-channel ZS values from DB, set useConfigZSvalues = cms.int32(0)
# the deafult uses the ZS threshold values from the DB
import FWCore.ParameterSet.Config as cms



simHcalDigis = cms.EDFilter("HcalRealisticZS",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    mode = cms.int32(0),
    markAndPass = cms.bool(False),
    HBlevel = cms.int32(8),
    HElevel = cms.int32(9),
    HOlevel = cms.int32(8),
    HFlevel = cms.int32(10),
    useConfigZSvalues = cms.int32(0)
)



