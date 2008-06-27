# The following comments couldn't be translated into the new config version:

#,
#  service = Timing{ }

import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# run simulation, with EcalHits Validation specific watcher 
process.load("SimG4Core.Application.g4SimHits_cfi")

#  replace g4SimHits.Watchers = {
#       { string type = "EcalSimHitsValidProducer"
#         untracked string instanceLabel="EcalValidInfo"
#         untracked bool verbose = false
#       }
#  }
# Mixing Module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

# ECAL digitization sequence
process.load("SimCalorimetry.Configuration.ecalDigiSequence_cff")

# ECAL digis validation sequence
#include "Validation/EcalDigis/data/ecalDigisValidationSequence.cff"
# Defines Ecal seletive readout validation module, ecalSelectiveReadoutValidation:
process.load("Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/045D2CA8-91F0-DC11-AD70-001617DBD230.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/08CC42A1-91F0-DC11-B764-001617E30CC8.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/2806C658-A7F0-DC11-AFAF-001617E30E2C.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/46EF7A8F-91F0-DC11-AC0F-000423D9997E.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/48808541-C9F0-DC11-BCCB-001617C3B76E.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/8285DF94-91F0-DC11-9BC6-000423D98868.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/86609AE5-92F0-DC11-ACD0-000423D94700.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/F25EA8A5-91F0-DC11-87B0-000423D98A44.root', 
        '/store/relval/2008/3/12/RelVal-RelValQCD_Pt_30_50-1205351746/0000/F4FB978F-91F0-DC11-8AAB-000423D98B08.root')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('Photon_E30GeV_all_EcalValidation.root')
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.p1 = cms.Path(process.ecalSelectiveReadoutValidation)
process.outpath = cms.EndPath(process.o1)
process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.simEcalDigis.writeSrFlags = True

