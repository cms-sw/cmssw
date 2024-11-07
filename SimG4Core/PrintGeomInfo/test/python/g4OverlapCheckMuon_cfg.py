###############################################################################
# Way to use this:
#   cmsRun g4OverlapCheckMuon_cfg.py geometry=2021 tol=0.1
#
#   Options for geometry 2016, 2017, 2021, Run4D102, Run4D103, Run4D104,
#                        Run4D108, Run4D109, Run4D110, Run4D111, Run4D112, 
#                        Run4D113, Run4D114, Run4D115, Run4D116
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2021",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2016, 2017, 2021, Run4D102, Run4D103, Run4D104, Run4D108, Run4D109, Run4D110, Run4D111, Run4D112, Run4D113, Run4D114, Run4D115, Run4D116")
options.register('tol',
                 0.01,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0"
)

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "Run4D102"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D102Reco_cff')
    baseName = 'MuonRun4D102'
elif (options.geometry == "Run4D103"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D103Reco_cff')
    baseName = 'MuonRun4D103'
elif (options.geometry == "Run4D104"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('G4PrintGeometry',Phase2C22I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D104Reco_cff')
    baseName = 'MuonRun4D104'
elif (options.geometry == "Run4D108"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D108Reco_cff')
    baseName = 'MuonRun4D108'
elif (options.geometry == "Run4D109"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('G4PrintGeometry',Phase2C22I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D109Reco_cff')
    baseName = 'MuonRun4D109'
elif (options.geometry == "Run4D110"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
    baseName = 'MuonRun4D110'
elif (options.geometry == "Run4D111"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('G4PrintGeometry',Phase2C22I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D111Reco_cff')
    baseName = 'MuonRun4D111'
elif (options.geometry == "Run4D112"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('G4PrintGeometry',Phase2C22I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D112Reco_cff')
    baseName = 'MuonRun4D112'
elif (options.geometry == "Run4D113"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('G4PrintGeometry',Phase2C22I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D113Reco_cff')
    baseName = 'MuonRun4D113'
elif (options.geometry == "Run4D114"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D114Reco_cff')
    baseName = 'MuonRun4D114'
elif (options.geometry == "Run4D115"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('G4PrintGeometry',Phase2C20I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D115Reco_cff')
    baseName = 'MuonRun4D115'
elif (options.geometry == "Run4D116"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    process.load('Configuration.Geometry.GeometryExtendedRun4D116Reco_cff')
    baseName = 'MuonRun4D116'
elif (options.geometry == "2016"):
    from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
    process = cms.Process('G4PrintGeometry',Run2_2016)
    process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
    baseName = 'Muon2016'
elif (options.geometry == "2017"):
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process('G4PrintGeometry',Run2_2017)
    process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
    baseName = 'Muon2017'
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('G4PrintGeometry',Run3_DDD)
    process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
    baseName = 'Muon2021'

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
process.g4SimHits.G4CheckOverlap.RegionFlag = cms.bool(False)
# list of names
process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('OCMS')
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
