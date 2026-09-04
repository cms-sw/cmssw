# Geant4 step trees for the BL-fit material map (BLMaterialMap<tag>.cc).
#
# Shoots straight, non-interacting rays (10 GeV muon neutrinos under DummyPhysics with the magnetic
# field off) from the nominal interaction point through the full Geant4 detector of the chosen
# geometry and records every step inside the Tracker and beam-pipe volumes with the release's own
# MaterialBudgetAction watcher (AllStepsToTree). blMaterialMapBuild turns the trees into the (r,z)
# lattice and blMaterialMapEmit.py writes the table; blMaterialMapRun.sh drives the three steps.
#
#   cmsRun blMaterialMapRays_cfg.py nEvents=150000 seed=1 out=rays_001.root \
#          geometry=Configuration.Geometry.GeometryExtendedRun4D121Reco_cff era=Phase2C22I13M9
import importlib
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('nEvents', 150000, VarParsing.multiplicity.singleton, VarParsing.varType.int, "rays")
options.register('seed', 1, VarParsing.multiplicity.singleton, VarParsing.varType.int, "job seed (1-based)")
options.register('etaMin', -6.0, VarParsing.multiplicity.singleton, VarParsing.varType.float, "eta min")
options.register('etaMax', 6.0, VarParsing.multiplicity.singleton, VarParsing.varType.float, "eta max")
options.register('out', 'blMaterialMapRays.root', VarParsing.multiplicity.singleton, VarParsing.varType.string,
                 "step-tree file")
options.register('geometry', 'Configuration.Geometry.GeometryExtendedRun4D121Reco_cff',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, "geometry configuration")
options.register('era', 'Phase2C22I13M9', VarParsing.multiplicity.singleton, VarParsing.varType.string, "era")
options.parseArguments()

era = getattr(importlib.import_module('Configuration.Eras.Era_%s_cff' % options.era), options.era)
process = cms.Process("BLMATERIALMAP", era)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load(options.geometry)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("IOMC.RandomEngine.IOMC_cff")
process.load("GeneratorInterface.Core.generatorSmeared_cfi")
from Configuration.StandardSequences.VtxSmeared import VtxSmeared
process.load(VtxSmeared['NoSmear'])  # every ray starts exactly at (0,0,0)

process.RandomNumberGeneratorService.generator.initialSeed = 1000 + options.seed
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9000 + options.seed
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 5000 + options.seed

process.source = cms.Source("EmptySource", firstRun=cms.untracked.uint32(1), firstEvent=cms.untracked.uint32(1))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.nEvents))

process.MessageLogger = cms.Service("MessageLogger",
    cerr=cms.untracked.PSet(enable=cms.untracked.bool(False)),
    cout=cms.untracked.PSet(enable=cms.untracked.bool(True),
                            threshold=cms.untracked.string('WARNING'),
                            default=cms.untracked.PSet(limit=cms.untracked.int32(10))))

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters=cms.PSet(
        PartID=cms.vint32(14),
        MinEta=cms.double(options.etaMin), MaxEta=cms.double(options.etaMax),
        MinPhi=cms.double(-3.14159265359), MaxPhi=cms.double(3.14159265359),
        MinE=cms.double(10.0), MaxE=cms.double(10.0)),
    AddAntiParticle=cms.bool(False), Verbosity=cms.untracked.int32(0))

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.StackingAction.TrackNeutrino = cms.bool(True)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type=cms.string('MaterialBudgetAction'),
    MaterialBudgetAction=cms.PSet(
        HistosFile=cms.string('None'),
        AllStepsToTree=cms.bool(True),
        HistogramList=cms.string('None'),
        SelectedVolumes=cms.vstring('Tracker', 'BEAM'),
        TreeFile=cms.string(options.out),
        StopAfterProcess=cms.string('None'),
        TextFile=cms.string('None'),
        storeDecay=cms.untracked.bool(False),
        EminDecayProd=cms.untracked.double(0.0))))

process.p1 = cms.Path(process.generator * process.VtxSmeared * process.generatorSmeared * process.g4SimHits)
