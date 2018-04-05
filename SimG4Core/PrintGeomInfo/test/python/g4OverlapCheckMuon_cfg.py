import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load('Configuration.Geometry.GeometryIdeal_cff')
#process.load('Configuration.Geometry.GeometryExtended_cff')
#process.load('Configuration.Geometry.GeometryExtended2015_cff')
#process.load('Configuration.Geometry.GeometryExtended2017_cff')
#process.load('Configuration.Geometry.GeometryExtended2019_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D13_cff')
process.load('SimG4Core.PrintGeomInfo.cmsMuon2017XML_cfi')
#process.load('SimG4Core.PrintGeomInfo.cmsBeamPipe2017XML_cfi')

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

process.MessageLogger.destinations = cms.untracked.vstring("muon2017.overlaps")

# enable Geant4 overlap check 
process.g4SimHits.CheckOverlap = True

# Geant4 overlap check conditions 
process.g4SimHits.G4CheckOverlap.Tolerance  = cms.untracked.double(0.0)
process.g4SimHits.G4CheckOverlap.Resolution = cms.untracked.int32(10000)
# tells if NodeName is G4Region or G4PhysicalVolume
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.untracked.bool(False)
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('OCMS')
# enable dump gdml file 
process.g4SimHits.G4CheckOverlap.gdmlFlag   = cms.untracked.bool(False)
# if defined a G4PhysicsVolume info is printed
process.g4SimHits.G4CheckOverlap.PVname     = ''
# if defined a list of daughter volumes is printed
process.g4SimHits.G4CheckOverlap.LVname     = ''

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#
