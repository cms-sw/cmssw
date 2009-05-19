# The following comments couldn't be translated into the new config version:

# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
import FWCore.ParameterSet.Config as cms

simHcalDigis = cms.EDFilter("HcalRealisticZS",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    mode = cms.int32(0),
    HBlevel = cms.int32(8),
    HElevel = cms.int32(9),
    HOlevel = cms.int32(8),
    HFlevel = cms.int32(10)
)



