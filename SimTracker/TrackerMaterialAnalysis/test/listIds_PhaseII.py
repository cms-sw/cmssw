#! /usr/bin/env cmsRun
# cmsRun listIds_PhaseII.py fromDB=False

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("MaterialAnalyser")

options = VarParsing('analysis')

options.register('fromDB',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 'Read Geometry from DB?',
)

options.parseArguments()

if options.fromDB :
   process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
   from Configuration.AlCa.GlobalTag import GlobalTag
   process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')
else:
   process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
   process.trackerGeometry.applyAlignment = False # needed to avoid to pass the Global Position Record

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


