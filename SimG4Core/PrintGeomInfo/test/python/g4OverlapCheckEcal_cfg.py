###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheckEcal_cfg.py geometry=2021 tol=0.1
#
#   Options for geometry 2016, 2017, 2021, 2026D88, 2026D92, 2026D93, 2026D99
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2021",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2016, 2017, 2021, 2026D88, 2026D92, 2026D93, 2026D99")
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

if (options.geometry == "2026D88"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    baseName = 'Ecal2026D88'
elif (options.geometry == "2026D92"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    baseName = 'Ecal2026D92'
elif (options.geometry == "2026D93"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
    baseName = 'Ecal2026D93'
elif (options.geometry == "2026D99"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtended2026D99Reco_cff')
    baseName = 'Ecal2026D99'
elif (options.geometry == "2016"):
    from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
    process = cms.Process('G4PrintGeometry',Run2_2016)
    process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
    baseName = 'Ecal2016'
elif (options.geometry == "2017"):
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process('G4PrintGeometry',Run2_2017)
    process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
    baseName = 'Ecal2017'
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('G4PrintGeometry',Run3_DDD)
    process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
    baseName = 'Ecal2021'

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

