#!/usr/bin/env cmsRun

# cmsRun trackingMaterialProducer.py nEvents=1000 fromDB=False

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("Geometry")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.files.debugTrackingMaterialProducer = dict()
process.MessageLogger.TrackingMaterialProducer=dict()

process.load('Configuration.EventContent.EventContent_cff')

## MC Related stuff
process.load('Configuration.StandardSequences.Generator_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')

### Loading 10GeV neutrino gun generator
process.load("SimTracker.TrackerMaterialAnalysis.single10GeVNeutrino_cfi")

### Load vertex generator w/o smearing
from Configuration.StandardSequences.VtxSmeared import VtxSmeared
process.load(VtxSmeared['NoSmear'])

# detector simulation (Geant4-based) with tracking material accounting 
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialProducer_cff")
#For some reason now neutrino are no longer tracked, so we need to force it.
process.trackingMaterialProducer.StackingAction.TrackNeutrino = True

options = VarParsing('analysis')

options.register('nEvents',
                 200000,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Maximum number of events"
)

options.register('fromDB',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 "Read from Geometry DB?",
)

options.parseArguments()

if options.fromDB :
   process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
   from Configuration.AlCa.GlobalTag import GlobalTag
   process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
else:
   process.load('Configuration.Geometry.GeometryExtended2026D49_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)

# Input source
process.source = cms.Source("EmptySource")

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',                                                       # drop all objects
        'keep MaterialAccountingTracks_trackingMaterialProducer_*_*'),  # but the material accounting information
    fileName = cms.untracked.string('file:material.root')
)

process.path = cms.Path(process.generator
                        * process.VtxSmeared
                        * process.generatorSmeared
                        * process.trackingMaterialProducer)
process.outpath = cms.EndPath(process.out)

