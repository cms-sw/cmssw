###############################################################################
# Way to use this:
#   cmsRun grunPrintG4Solids_cfg.py geometry=D98 dd4hep=False
#
#   Options for geometry D88, D91, D92, D93, D94, D95, D96, D98, D99, D100,
#                        D101
#   Options for type DDD, DD4hep
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
                  "geometry of operations: D88, D91, D92, D93, D94, D95, D96, D98, D99, D100, D101")
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

if (options.type == "DD4hep"):
    geomFile = "Configuration.Geometry.GeometryDD4hepExtended2026" + options.geometry + "Reco_cff"
    if (options.geometry == "D94"):
        from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
        process = cms.Process('PrintG4Solids',Phase2C20I13M9,dd4hep)
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('PrintG4Solids',Phase2C17I13M9,dd4hep)
else:
    geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
    if (options.geometry == "D94"):
        from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
        process = cms.Process('PrintG4Solids',Phase2C20I13M9)
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('PrintG4Solids',Phase2C17I13M9)

print("Geometry file Name: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.HepPDTESSource.pdt_cfi')
process.load('IOMC.RandomEngine.IOMC_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimG4Core.Application.g4SimHits_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

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

if (options.type == "DD4hep"):
    dd4hep = True
else:
    dd4hep = False

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    dd4hep         = cms.untracked.bool(dd4hep),
    dumpVolumes    = cms.untracked.vstring(),
    dumpShapes     = cms.untracked.vstring("G4ExtrudedSolid"),
    type           = cms.string('PrintG4Solids')
))

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)
