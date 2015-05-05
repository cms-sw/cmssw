#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialGroups_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.listGroups = cms.EDAnalyzer("ListGroups",
                                    SaveSummaryPlot = cms.untracked.bool(True))
process.path = cms.Path(process.listGroups)
