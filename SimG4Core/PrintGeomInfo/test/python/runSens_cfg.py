###############################################################################
# Way to use this:
#   cmsRun runSens_cfg.py geometry=Run3
#
#   Options for geometry Run3, D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Run3",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: Run3, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

if (options.geometry == "Run3"):
    geomFile = "Configuration.Geometry.GeometryExtended2021Reco_cff"
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('PrintSensitive',Run3_DDD)
else:
    geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PrintSensitive',Phase2C11M9)

print("Geometry file: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger.G4cout=dict()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load('SimGeneral.HepPDTESSource.pdt_cfi')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.generatorSmeared_cfi')

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(-2.5),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt  = cms.double(9.99),
        MaxPt  = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
         initialSeed = cms.untracked.uint32(123456789),
         engineName = cms.untracked.string('HepJamesRandom')
    ),
    VtxSmeared = cms.PSet(
        engineName = cms.untracked.string('HepJamesRandom'),
        initialSeed = cms.untracked.uint32(98765432)
    ),
    g4SimHits = cms.PSet(
         initialSeed = cms.untracked.uint32(11),
         engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.load('SimG4Core.Application.g4SimHits_cfi')

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)

process.g4SimHits.UseMagneticField        = False
process.g4SimHits.Physics.DefaultCutValue = 10. 
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
	Name           = cms.string('*'),
        DD4Hep         = cms.bool(False),
	type           = cms.string('PrintSensitive')
))
