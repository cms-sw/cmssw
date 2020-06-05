#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")

process.DDDetectorESProducer = cms.ESSource(
    "DDDetectorESProducer",
    confGeomXMLFiles = cms.FileInPath('SimTracker/TrackerMaterialAnalysis/data/dd4hep_trackingMaterialGroups.xml'),
    appendToDataLabel = cms.string('CMS')
)

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkT14_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.HGCalCommonData.hgcalParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *

process.load('FWCore.MessageService.MessageLogger_cfi')

process.DDCompactViewESProducer = cms.ESProducer(
  "DDCompactViewESProducer",
  appendToDataLabel = cms.string('CMS')
)

# Analyze and plot the tracking material
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_ForPhaseII_dd4hep_cff")
process.trackingMaterialAnalyser.SplitMode         = "NearestLayer"
process.trackingMaterialAnalyser.SaveParameters    = True
process.trackingMaterialAnalyser.SaveXML           = True
process.trackingMaterialAnalyser.SaveDetailedPlots = False

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:material.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger.destinations.extend(["LogTrackingMaterialAnalysis"])
process.MessageLogger.categories.append("TrackingMaterialAnalysis")
process.path = cms.Path(process.trackingMaterialAnalyser)

