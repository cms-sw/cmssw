#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.destinations.extend(["LogTrackingMaterialAnalysis"])
process.MessageLogger.categories.append("TrackingMaterialAnalysis")

# Add our custom detector grouping to DDD
process.XMLIdealGeometryESSource.geomXMLFiles.extend(['SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups_ForPhaseII.xml'])

# Analyze and plot the tracking material
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_ForPhaseII_cff")
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


