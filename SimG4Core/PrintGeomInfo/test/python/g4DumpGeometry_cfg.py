import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.Geometry.GeometryExtended2017_cff')
#process.load('Configuration.Geometry.GeometryExtended2018_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D45_cff')

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

process.MessageLogger.cerr.enable = False
process.MessageLogger.files.Ecal2017 = dict(extension ="info")

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = cms.bool(True)

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = cms.string("2017")
#process.g4SimHits.G4CheckOverlap.OutputBaseName = cms.string("2026D45")
process.g4SimHits.G4CheckOverlap.OverlapFlag = cms.bool(False)
process.g4SimHits.G4CheckOverlap.Tolerance  = cms.double(0.0)
process.g4SimHits.G4CheckOverlap.Resolution = cms.int32(10000)
# tells if NodeName is G4Region or G4PhysicalVolume
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.bool(True)
# list of region names for which overlap check is performed
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('EcalRegion')
# enable dump gdml file 
process.g4SimHits.G4CheckOverlap.gdmlFlag   = cms.bool(False)
# if defined a G4PhysicsVolume info is printed
process.g4SimHits.G4CheckOverlap.PVname     = ''
# if defined a list of daughter volumes is printed
process.g4SimHits.G4CheckOverlap.LVname     = 'ECAL'

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#
