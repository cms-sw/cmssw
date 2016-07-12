import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2015devXML_cfi")
#process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.Geometry.GeometryExtended2017_cff')

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

process.MessageLogger.destinations = cms.untracked.vstring("QuadRegion.txt")

# enable Geant4 overlap check 
process.g4SimHits.CheckOverlap = True

# Geant4 overlap check volume/conditions 
process.g4SimHits.G4CheckOverlap.Tolerance  = cms.untracked.double(0.0)
process.g4SimHits.G4CheckOverlap.Resolution = cms.untracked.int32(10000)
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.untracked.bool(True)
process.g4SimHits.G4CheckOverlap.gdmlFlag   = cms.untracked.bool(True)
process.g4SimHits.G4CheckOverlap.PVname     = ''
process.g4SimHits.G4CheckOverlap.LVname     = ''
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('QuadRegion')

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#
