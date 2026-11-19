###############################################################################
# Way to use this:
#   cmsRun runP_HCALP2_dd4hep_cfg.py geometry=D126
#
#   Options for geometry D104, D110, D111, D112, D113, D114, D115, D120, D121
#                        D122, D123, D124, D125, D126, D127, D128
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D127",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D104, D110, D111, D112, D113, D114, D123, D124, D125, D126, D127, D128")
### get and parse the command line arguments
options.parseArguments()
print(options)

####################################################################
# Use the options

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryDD4hepExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('PROD',ERA,dd4hep)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load(geomFile)
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
if hasattr(process,'MessageLogger'):
    process.MessageLogger.MaterialBudget=dict()

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('matbdgHCAL_run4.root')
)

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.LHCTransport = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcal = cms.PSet(
        FillHisto    = cms.untracked.bool(True),
        PrintSummary = cms.untracked.bool(True),
        DoHCAL       = cms.untracked.bool(True),
        TestVolumes  = cms.untracked.bool(True),
        NBinPhi      = cms.untracked.int32(180),
        NBinEta      = cms.untracked.int32(260),
        MaxEta       = cms.untracked.double(5.2),
        EtaLow       = cms.untracked.double(-5.2),
        EtaHigh      = cms.untracked.double(5.2),
        EtaMinP      = cms.untracked.double(5.2),
        EtaMaxP      = cms.untracked.double(0.0),
        EtaLowMin    = cms.untracked.double(0.783),
        EtaLowMax    = cms.untracked.double(0.870),
        EtaMidMin    = cms.untracked.double(2.650),
        EtaMidMax    = cms.untracked.double(2.868),
        EtaHighMin   = cms.untracked.double(2.868),
        EtaHighMax   = cms.untracked.double(3.000),
        RMax         = cms.untracked.double(5.0),
        ZMax         = cms.untracked.double(14.0),
        Fromdd4hep   = cms.untracked.bool(False)
    ),
    type = cms.string('MaterialBudgetHcal')
))


