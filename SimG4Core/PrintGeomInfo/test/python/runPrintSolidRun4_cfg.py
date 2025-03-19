#######################################################9########################
# Way to use this:
#   cmsRun runPrintSolidRun4_cfg.py type=DDD geometry=D110
#
#   Options for type DDD, DD4hep
#   Options for geometry D95, D96, D98, D99, D100, D101, D102, D103, D104,
#                        D105, D106, D107, D108, D109, D110, D111, D112, D113,
#                        D114, D115, D116, D117, D118, D119
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
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D95, D96, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115, D116, D117, D118, D119")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

geomName = "Run4" + options.geometry
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry Name:   ", geomName)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

if (options.type == "DD4hep"):
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process('G4PrintGeometry',ERA,dd4hep)
    geomFile = "Configuration.Geometry.GeometryDD4hepExtended" + geomName + "Reco_cff"
else:
    process = cms.Process('G4PrintGeometry',ERA)
    geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"

print("Geometry file Name: ", geomFile)

process.load(geomFile)
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
