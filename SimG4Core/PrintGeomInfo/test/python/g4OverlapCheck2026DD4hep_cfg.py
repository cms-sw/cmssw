###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheck2026DD4hep_cfg.py geometry=D92 tol=0.1
#
#   Options for geometry D77, D83, D88, D92
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
                  "geometry of operations: D77, D83, D88, D92")
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

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9,Run3_dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D83Reco_cff')
    baseName = 'cms2026D83DD4hep'
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11,Run3_dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D77Reco_cff')
    baseName = 'cms2026D77DD4hep'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('PROD',Phase2C11,Run3_dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D92Reco_cff')
    baseName = 'cms2026D92DD4hep'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PROD',Phase2C11M9,Run3_dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D88Reco_cff')
    baseName = 'cms2026D88DD4hep'

print("Base file Name: ", baseName)

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
