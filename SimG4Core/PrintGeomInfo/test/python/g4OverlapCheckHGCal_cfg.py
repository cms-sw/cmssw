###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheckHGCal_cfg.py geometry=D86 tol=0.1
#
#   Options for geometry D77, D83, D86, D88
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D86",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D77, D83, D86, D88")
options.register('tol',
                 0.1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0"
)

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import PHase2c11m9
    process = cms.Process('G4PrintGeometry',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    baseName = 'HGCal2026D83'
elif (options.geometry == "D86"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D86Reco_cff')
    baseName = 'HGCal2026D86'
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('G4PrintGeometry',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    baseName = 'HGCal2026D77'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    baseName = 'HGCal2026D86'

print("Base file Name: ", baseName)

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
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.bool(True)
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('HGCalRegion')
# enable dump gdml file 
process.g4SimHits.G4CheckOverlap.gdmlFlag   = cms.bool(True)
# if defined a G4PhysicsVolume info is printed
process.g4SimHits.G4CheckOverlap.PVname     = ''
# if defined a list of daughter volumes is printed
process.g4SimHits.G4CheckOverlap.LVname     = ''

# extra output files, created if a name is not empty
process.g4SimHits.FileNameField   = ''
process.g4SimHits.FileNameGDML    = ''
process.g4SimHits.FileNameRegions = ''
#

#process.load('Geometry.HGCalCommonData.testHGCalV10XML_cfi')
