#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

# Analyze and plot the tracking material
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:material.root')
)

process.path = cms.Path(process.trackingMaterialAnalyser)
