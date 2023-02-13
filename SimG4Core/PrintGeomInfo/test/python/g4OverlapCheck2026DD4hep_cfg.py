###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheck2026DD4hep_cfg.py geometry=D92 tol=0.1
#
#   Options for geometry D86, D88, D91, D92, D93, D94, D95, D96, D98, D99
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D86, D88, D91, D92, D93, D94, D95, D96, D98, D99")
options.register('tol',
                 0.1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.01, 0.1, 1.0"
)

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
if (options.geometry == "D94"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('OverlapCheck',Phase2C20I13M9,dd4hep)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('OverlapCheck',Phase2C17I13M9,dd4hep)

geomFile = "Configuration.Geometry.GeometryDD4hepExtended2026" + options.geometry + "Reco_cff"
baseName = "cms2026" + options.geometry + "DD4hep"

print("Geometry file Name: ", geomFile)
print("Base file Name:     ", baseName)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

#if hasattr(process,'MessageLogger'):
#    process.MessageLogger.HGCalGeom=dict()

from SimG4Core.PrintGeomInfo.g4TestGeometry_cfi import *
process = checkOverlap(process)

# enable Geant4 overlap check 
process.g4SimHits.CheckGeometry = True

# Geant4 geometry check 
process.g4SimHits.G4CheckOverlap.OutputBaseName = cms.string(baseName)
process.g4SimHits.G4CheckOverlap.OverlapFlag = cms.bool(True)
process.g4SimHits.G4CheckOverlap.Tolerance  = cms.double(options.tol)
process.g4SimHits.G4CheckOverlap.Resolution = cms.int32(10000)
process.g4SimHits.G4CheckOverlap.Depth      = cms.int32(-1)
# tells if NodeName is G4Region or G4PhysicalVolume
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.bool(False)
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('cms:OCMS_1')
# enable dump gdml file 
process.g4SimHits.G4CheckOverlap.gdmlFlag   = cms.bool(False)
# if defined a G4PhysicsVolume info is printed
process.g4SimHits.G4CheckOverlap.PVname     = ''
# if defined a list of daughter volumes is printed
process.g4SimHits.G4CheckOverlap.LVname     = ''

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#
