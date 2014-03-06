#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

# Configuration and Conditions
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

# Analyze and plot the tracking material
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_cff")
process.trackingMaterialAnalyser.SplitMode         = "NearestLayer"
process.trackingMaterialAnalyser.SaveParameters    = True
process.trackingMaterialAnalyser.SaveXML           = True
process.trackingMaterialAnalyser.SaveDetailedPlots = True
  
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:material.root')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.path = cms.Path(process.trackingMaterialAnalyser)
