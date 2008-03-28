import FWCore.ParameterSet.Config as cms

# Trigger Primitive Producer
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
# esmodule creating  records + corresponding empty essource
EcalTrigPrimESProducer = cms.ESProducer("EcalTrigPrimESProducer",
    DatabaseFileEE = cms.untracked.string('TPG_EE.txt'),
    DatabaseFileEB = cms.untracked.string('TPG_EB.txt')
)

tpparams = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLinearizationConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams2 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPedestalsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams3 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGSlidingWindowRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams4 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams5 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams6 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLutGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams7 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLutIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams8 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainEBIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams9 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainEBGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams10 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainStripEERcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

tpparams11 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainTowerEERcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


