#! /usr/bin/env cmsRun

import sys
import FWCore.ParameterSet.Config as cms

from SimTracker.TrackerMaterialAnalysis.trackingMaterialVarParsing import options

process = cms.Process("MaterialAnalyser")

if options.geometry == 'run2':
    process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
    # Add our custom detector grouping to DDD
    process.XMLIdealGeometryESSource.geomXMLFiles.extend(['SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups.xml'])
elif options.geometry == 'Phase1':
    process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
    # Add our custom detector grouping to DDD
    process.XMLIdealGeometryESSource.geomXMLFiles.extend(['SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups_ForPhaseI.xml'])
elif options.geometry == 'Phase2':
    process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
    # Add our custom detector grouping to DDD
    process.XMLIdealGeometryESSource.geomXMLFiles.extend(['SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups_ForPhaseII_D4.xml'])
else:
    print("Unknow geometry, quitting.")
    sys.exit(1)

process.load("Configuration.StandardSequences.MagneticField_cff")


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.listGroups = cms.EDAnalyzer("ListGroups",
                                    SaveSummaryPlot = cms.untracked.bool(True))
process.path = cms.Path(process.listGroups)
