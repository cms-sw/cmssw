import FWCore.ParameterSet.Config as cms

# esmodule creating  records + corresponding empty essource
EcalTrigPrimSpikeESProducer = cms.ESProducer("EcalTrigPrimSpikeESProducer",
    TCCZeroingThreshold = cms.untracked.uint32(1023)
)

tpspikeparms = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGSpikeRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

