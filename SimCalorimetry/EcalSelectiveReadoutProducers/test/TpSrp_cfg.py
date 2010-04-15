# The following comments couldn't be translated into the new config version:

#  module o1 = PoolOutputModule { 
#    untracked string fileName = "srp_validation_in.root"
#
#    untracked vstring outputCommands = 
#    {
#     "keep *"
#    }
#  }

import FWCore.ParameterSet.Config as cms

process = cms.Process("TpSrp")

#Geometry
#
#include "Geometry/CMSCommonData/data/cmsSimIdealGeometryXML.cfi"
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

process.load("SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(1),
    fileNames = cms.untracked.vstring('file://srp_validation_in.root') ##srp_validation_in.root'}

)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(135799753)
)

process.p1 = cms.Path(process.simEcalDigis)
process.simEcalDigis.dumpFlags = 10

