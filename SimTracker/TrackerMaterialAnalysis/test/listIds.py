#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.listIds = cms.EDAnalyzer("ListIds")
process.path = cms.Path(process.listIds)
