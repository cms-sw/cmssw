import FWCore.ParameterSet.Config as cms

# TP Emulator Producer:
from CalibCalorimetry.Configuration.Ecal_FakeConditions_cff import *


# Ecal Trig Prim module
simEcalTriggerPrimitiveDigis = cms.EDProducer("EcalTrigPrimProducer",
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



