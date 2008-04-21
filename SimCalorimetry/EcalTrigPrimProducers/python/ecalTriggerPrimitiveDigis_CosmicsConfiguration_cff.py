import FWCore.ParameterSet.Config as cms

# TP Emulator Producer:
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from CalibCalorimetry.Configuration.Ecal_FakeConditions_cff import *
# Sources of record
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

tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

# Event Setup module
EcalTrigPrimESProducer = cms.ESProducer("EcalTrigPrimESProducer",
    DatabaseFile = cms.untracked.string('TPG_MIPs.txt')
)

# Ecal Trig Prim module
ecalTriggerPrimitiveDigis = cms.EDProducer("EcalTrigPrimProducer",
    BarrelOnly = cms.bool(True),
    TTFHighEnergyEB = cms.double(1.0),
    InstanceEB = cms.string('ebDigis'),
    InstanceEE = cms.string(''),
    TTFHighEnergyEE = cms.double(1.0),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10

    Famos = cms.bool(False),
    TTFLowEnergyEE = cms.double(1.0),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Label = cms.string('ecalEBunpacker'),
    TTFLowEnergyEB = cms.double(1.0) ## this + the following is added from 140_pre4 on

)


