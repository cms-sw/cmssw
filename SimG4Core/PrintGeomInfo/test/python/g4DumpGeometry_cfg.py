###############################################################################
# Way to use this:
#   cmsRun g4DumpGeometry_cfg.py geometry=Run4D110
#
#   Options for geometry 2015, 2017, 2018, Run4D110, Run4D116
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Run4D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2015, 2017, 2018, Run4D110, Run4D116")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
outFile = options.geometry
print("Geometry file Name:    ", geomFile)
print("Output Base file Name: ", outFile)

process = cms.Process("G4PrintGeometry")

process.load(geomFile)

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = cms.bool(True)

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = cms.string(outFile)
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
