###############################################################################
# Way to use this:
#   cmsRun testHGCalGuardRing_cfg.py geometry=D110 type=DDD
#
#   Options for geometry: D104, D110, D116, D120
#               type: DDD, DD4hep
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re, random
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D104, D110, D116, D120")
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

geomName = "Run4" + options.geometry
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

if (options.type == "DD4hep"):
    from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
    process = cms.Process('GuardRing',ERA,dd4hep)
    geomFile = "Configuration.Geometry.Geometry" + options.type +"Extended" + geomName + "Reco_cff"
else:
    process = cms.Process('GuardRing',ERA)
    geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"

print("Geometry file:   ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBAL_TAG, '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCSim=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

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

process.load("SimG4CMS.Calo.hgcalTestGuardRing_cff")

process.p1 = cms.Path(process.generator*process.hgcalTestGuardRingEE*process.hgcalTestGuardRingHE)
