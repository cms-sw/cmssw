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

#Conditions:
#process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_38Y_V1::All'

#----------------------------------------------------------------------
#To overwrite Selective readout settings with settings from a local DB (sqllite file): 
# process.GlobalTag.toGet = cms.VPSet(
#      cms.PSet(record = cms.string("EcalSRSettingsRcd"),
#                          tag = cms.string("EcalSRSettings_v00_beam10_mc"),
#                          connect = cms.untracked.string("sqlite_file:EcalSRSettings_v00_beam10_mc.db")
#               )
#      )
#----------------------------------------------------------------------

process.load("SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file://srp_validation_in.root') ##srp_validation_in.root'}
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(135799753)
)

# Defines Ecal seletive readout validation module, ecalSelectiveReadoutValidation:
process.load("Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi")
process.ecalSelectiveReadoutValidation.outputFile = 'srvalid_hists.root'
process.ecalSelectiveReadoutValidation.ecalDccZs1stSample = 3
process.ecalSelectiveReadoutValidation.dccWeights = [ -1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266 ]
process.ecalSelectiveReadoutValidation.histDir = ''
process.ecalSelectiveReadoutValidation.histograms = [ 'all' ]

# DQM services
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''


process.p1 = cms.Path(process.simEcalDigis*process.ecalSelectiveReadoutValidation)
process.simEcalDigis.dumpFlags = 10

