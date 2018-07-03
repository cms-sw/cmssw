#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

readGeometryFromDB = False

if not readGeometryFromDB:
  process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
else:
# GlobalTag and geometry via GT
  process.load('Configuration.Geometry.GeometrySimDB_cff')
  process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
  from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
  process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
        firstRun = cms.untracked.uint32(1)
)

process.listIds = cms.EDAnalyzer("ListIds",
                                materials = cms.untracked.vstring("materials:Silicon", "tracker:tkLayout_SenSi"),
                                printMaterial = cms.untracked.bool(True)
                                )
process.path = cms.Path(process.listIds)


