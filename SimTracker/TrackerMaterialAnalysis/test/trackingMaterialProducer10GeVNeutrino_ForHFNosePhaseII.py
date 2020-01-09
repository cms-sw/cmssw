#!/usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("Geometry")

readGeometryFromDB = False

# N.B. for the time being we load the geometry from local
# XML, whle in future we will have to use the DB. This is
# only a temporary hack, since the material description has
# been updated in release via XML and the DB is behind.
if not readGeometryFromDB:
  process.load('Configuration.Geometry.GeometryExtended2026D44_cff')
else:
# GlobalTag and geometry via GT
  process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
  from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
  process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load('FWCore.MessageService.MessageLogger_cfi')
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
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialProducerHFNose_cff")
#For some reason now neutrino are no longer tracked, so we need to force it.
process.trackingMaterialProducer.StackingAction.TrackNeutrino = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200000)
)

# Input source
process.source = cms.Source("EmptySource")

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',                                                       # drop all objects
        'keep MaterialAccountingTracks_trackingMaterialProducer_*_*'),  # but the material accounting informations
    fileName = cms.untracked.string('file:material.root')
)

process.path = cms.Path(process.generator
                        * process.VtxSmeared
                        * process.generatorSmeared
                        * process.trackingMaterialProducer)
process.outpath = cms.EndPath(process.out)

def customizeMessageLogger(process):
    ### Easy customisation of MessageLogger ###
    # 1. Extend MessageLogger to monitor all modules: the * means any
    #    label for all defined python modules
    process.MessageLogger.debugModules.extend(['*'])
    # 2. Define destination and its default logging properties
    destination = 'debugTrackingMaterialProducer'
    how_to_debug = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
                                      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                      default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                      )
    # 3. Attach destination and its logging properties to the main process
    process.MessageLogger.destinations.extend([destination])
    process.MessageLogger._Parameterizable__addParameter(destination, how_to_debug)
    # 4. Define and extend the categories we would like to monitor
    log_debug_categories = ['TrackingMaterialProducer']
    process.MessageLogger.categories.extend(log_debug_categories)

    # 5. Extend the configuration of the configured destination so that it
    #    will trace all messages coming from the list of specified
    #    categories.
    unlimit_debug = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    for val in log_debug_categories:
        process.MessageLogger.debugTrackingMaterialProducer._Parameterizable__addParameter(val, unlimit_debug)

    return process

#process = customizeMessageLogger(process)
