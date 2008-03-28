# The following comments couldn't be translated into the new config version:

# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
# conservative choice
# 0=per channel, 1=triggerOR, 2=depthOR
import FWCore.ParameterSet.Config as cms

hcalDigis = cms.EDFilter("HcalSimpleAmplitudeZS",
    digiLabel = cms.InputTag("hcalUnsuppressedDigis"),
    hbhe = cms.PSet(
        firstSample = cms.int32(4),
        mode = cms.int32(2),
        samplesToAdd = cms.int32(2),
        twoSided = cms.bool(False),
        level = cms.int32(2)
    ),
    hf = cms.PSet(
        firstSample = cms.int32(3),
        mode = cms.int32(2),
        samplesToAdd = cms.int32(1),
        twoSided = cms.bool(False),
        level = cms.int32(2)
    ),
    ho = cms.PSet(
        firstSample = cms.int32(4),
        mode = cms.int32(0),
        samplesToAdd = cms.int32(2),
        twoSided = cms.bool(False),
        level = cms.int32(2)
    )
)


