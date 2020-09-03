#! /usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("MaterialAnalyser")


# Configuration and Conditions
# We cannot read the geometry from the DB, since we have to inject out custom-made
# material-budget grouping into the DDD of the detector. So we need to read the
# geometry using the XMLIdealGeometryRecord.
process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
#Global Tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Add our custom detector grouping to DDD
process.XMLIdealGeometryESSource.geomXMLFiles.extend(['SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups.xml'])

# Analyze and plot the tracking material
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_cff")
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

process.path = cms.Path(process.trackingMaterialAnalyser)


def customizeMessageLogger(process):
    ### Easy customisation of MessageLogger ###
    # 1. Extend MessageLogger to monitor all modules: the * means any
    #    label for all defined python modules
    process.MessageLogger.debugModules.extend(['*'])
    # 2. Define destination and its default logging properties
    destination = 'debugTrackingMaterialAnalyzer'
    how_to_debug = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
                                      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                      default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                      )
    # 3. Attach destination and its logging properties to the main process
    process.MessageLogger.destinations.extend([destination])
    process.MessageLogger._Parameterizable__addParameter(destination, how_to_debug)
    # 4. Define and extend the categories we would like to monitor
    log_debug_categories = ['TrackingMaterialAnalyser']
    process.MessageLogger.categories.extend(log_debug_categories)

    # 5. Extend the configuration of the configured destination so that it
    #    will trace all messages coming from the list of specified
    #    categories.
    unlimit_debug = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    for val in log_debug_categories:
        process.MessageLogger.debugTrackingMaterialAnalyzer._Parameterizable__addParameter(val, unlimit_debug)

    return process


process = customizeMessageLogger(process)
