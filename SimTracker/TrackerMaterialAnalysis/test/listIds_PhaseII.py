#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')

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


