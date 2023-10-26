#######################################################9########################
# Way to use this:
#   cmsRun runPrintSolid2026_cfg.py type=DDD geometry=D98
#
#   Options for type DDD, DD4hep
#   Options for geometry D86, D88, D91, D92, D93, D95, D96, D97, D98, D99,
#                        D100, D101
#
################################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")
options.register('geometry',
                 "D92",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D86, D88, D91, D92, D93, D95, D96, D97, D98, D99, D100, D101")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

if (options.type == "DDD"):
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)
    geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
    process.load(geomFile)
else:
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9,dd4hep)
    geomFile = "Configuration.Geometry.GeometryDD4hepExtended2026" + options.geometry + "Reco_cff"
    process.load(geomFile)

print("Geometry file Name: ", geomFile)

process.load('SimGeneral.HepPDTESSource.pdt_cfi')

process.load('IOMC.RandomEngine.IOMC_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimG4Core.Application.g4SimHits_cfi')
process.load('SimG4Core.PrintGeomInfo.printGeomSolids_cff')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.PrintGeom=dict()

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MinEta = cms.double(-3.5),
        MaxEta = cms.double(3.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(9.99),
        MaxE   = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.DefaultCutValue = 10. 
process.g4SimHits.LHCTransport = False

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.printGeomSolids)
