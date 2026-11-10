#!/usr/bin/env cmsRun
# Quirk pair gun through g4SimHits with the Run 3 geometry.
# Input: HepMC2 file from makeQuirkPairHepMC.py
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from SimG4Core.CustomPhysics.Exotica_Quirk_SIM_cfi import customiseQuirk

import argparse
import math
import sys
parser = argparse.ArgumentParser(prog=sys.argv[0], description='Quirk pair gun SIM test')
parser.add_argument("--inputFile", type=str, default="file:quirkPair.hepmc", help="HepMC2 input")
parser.add_argument("--outputFile", type=str, default="quirkPair_SIM.root", help="output file")
parser.add_argument("--maxEvents", type=int, default=-1, help="number of events")
parser.add_argument("--threads", type=int, default=1, help="number of threads")
parser.add_argument("--mass", type=float, default=250., help="quirk mass in GeV")
parser.add_argument("--lambdaEV", type=float, default=math.sqrt(1000. * 1.973269804e-10) * 1.e6,
                    help="Lambda in eV (default: string tension 1000 MeV/mm)")
parser.add_argument("--pdgId", type=int, default=17, help="quirk PDG code")
parser.add_argument("--charge", type=float, default=-1., help="charge of the positive PDG code")
parser.add_argument("--debugStep", type=float, default=0., help="conservation printout step in mm")
parser.add_argument("--verbose", type=int, default=0, help="quirk verbosity")
parser.add_argument("--reference", action="store_true",
                    help="no string: PDG 17 as the charge-1 HIP of the HSCP CustomPhysics (Lambda -> 0 reference)")
parser.add_argument("--seed", type=int, default=0, help="g4SimHits random seed (0 = default)")
parser.add_argument("--keepStopped", action="store_true", help="keep quirks stopped by energy loss while the partner moves")
parser.add_argument("--dumpEvery", type=int, default=0, help="write every N-th quirk step to quirkSteps_<thread>.txt")
parser.add_argument("--saveSecondaries", action="store_true", help="store the first-level secondaries (delta rays)")
args = parser.parse_args()

process = cms.Process('SIM', Run3_2024)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_realistic', '')

process.source = cms.Source("MCFileSource",
    fileNames = cms.untracked.vstring(args.inputFile),
    firstLuminosityBlockForEachRun = cms.untracked.VLuminosityBlockID()
)
process.maxEvents.input = args.maxEvents
process.options.numberOfThreads = args.threads
process.options.numberOfStreams = 0

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.QuirkDebugWatcher = dict()
process.MessageLogger.SimG4CoreCustomPhysics = dict()

process.g4SimHits.HepMCProductLabel = cms.InputTag("source", "generator")
process.g4SimHits.LHCTransport = False
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag("source", "generator")
if args.reference:
    process.load("SimG4Core.CustomPhysics.CustomPhysics_cfi")
    process.customPhysicsSetup.particlesDef = 'Configuration/Generator/data/particles_HIP3_stau_%d_GeV.txt' % int(args.mass)
    process.g4SimHits.Physics = cms.PSet(process.g4SimHits.Physics, process.customPhysicsSetup)
    process.g4SimHits.Physics.type = 'SimG4Core/Physics/CustomPhysics'
else:
    customiseQuirk(process, args.mass, args.lambdaEV, pdgId=args.pdgId, charge=args.charge,
                   debugStep=args.debugStep, verbose=args.verbose,
                   dumpEvery=args.dumpEvery, keepStopped=args.keepStopped)
if args.saveSecondaries:
    process.g4SimHits.StackingAction.SaveFirstLevelSecondary = True
    process.g4SimHits.TrackingAction.PersistencyEmin = 0.0001  # GeV
if args.seed > 0:
    process.RandomNumberGeneratorService.g4SimHits.initialSeed = args.seed

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 'keep *_g4SimHits_*_*', 'keep *_source_*_*'),
    fileName = cms.untracked.string(args.outputFile)
)

process.simulation_step = cms.Path(process.psim)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.out_step = cms.EndPath(process.output)
process.schedule = cms.Schedule(process.simulation_step, process.endjob_step, process.out_step)
