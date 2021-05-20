import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load('Configuration.Geometry.GeometryIdeal_cff')
#process.load('Configuration.Geometry.GeometryExtended_cff')
#process.load('Configuration.Geometry.GeometryExtended2015_cff')
#process.load('Configuration.Geometry.GeometryExtended2017_cff')
process.load('Configuration.Geometry.GeometryExtended2021_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D83_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
