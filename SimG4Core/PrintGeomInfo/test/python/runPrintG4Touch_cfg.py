###############################################################################
# Way to use this:
#   cmsRun grunPrintG4Touch_cfg.py geometry=2021
#
#   Options for geometry 2016, 2017, 2018, 2021, 2023
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
                  "geometry of operations: 2016, 2017, 2018, 2021, 2023")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"

if (options.geometry == "2016"):
    from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
    process = cms.Process('PrintG4Touch',Run2_2016)
elif (options.geometry == "2017"):
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process('PrintG4Touch',Run2_2017)
elif (options.geometry == "2018"):
    from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
    process = cms.Process('PrintG4Touch',Run2_2018)
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('PrintG4Touch',Run3_DDD)

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

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    dd4hep         = cms.untracked.bool(False),
    verbosity      = cms.untracked.bool(False),
    type           = cms.string('PrintG4Touch')
))

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)
