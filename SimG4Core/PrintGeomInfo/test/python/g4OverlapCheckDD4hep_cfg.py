###############################################################################
# Way to use this:  
#   cmsRun g4OverlapCheckDD4hep_cfg.py geometry=2021 tol=0.1
#
#   Options for geometry 2017, 2018, 2021, 2023
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2021",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2017, 2018, 2021, 2023")
options.register('tol',
                 0.1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Tolerance for checking overlaps: 0.0, 0.01, 0.1, 1.0"
)
options.parseArguments()
print(options)

baseName = "cmsDD4hep" + options.geometry
geomName = "Configuration.Geometry.GeometryDD4hepExtended" + options.geometry + "Reco_cff"

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
if (options.geometry == "2017"):
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process('G4PrintGeometry',Run2_2017,dd4hep)
elif (options.geometry == "2018"):
    from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
    process = cms.Process('G4PrintGeometry',Run2_2018,dd4hep)
else:
    from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
    process = cms.Process('G4PrintGeometry',Run3_dd4hep)

print("Base file Name: ", baseName)
print("Geom file Name: ", geomName)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load(geomName)

#if hasattr(process,'MessageLogger'):
#    process.MessageLogger.HCalGeom=dict()

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
# process.g4SimHits.G4CheckOverlap.NodeNames  = cms.vstring('DefaultRegionForTheWorld')
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
