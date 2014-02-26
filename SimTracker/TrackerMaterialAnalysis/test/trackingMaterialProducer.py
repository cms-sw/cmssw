#!/usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("Geometry")

# gaussian Vertex Smearing
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

# detector simulation (Geant4-based) with tracking material accounting 
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("SimTracker.TrackerMaterialAnalysis.trackingMaterialProducer_cff")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(    # but FwkJob category - those unlimitted

            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('FwkJob'),
    destinations = cms.untracked.vstring('cout')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("MCFileSource",
    # from http://cmsdoc.cern.ch/cms/data/CMSSW/Validation/Geometry/data/single_neutrino.random.dat
    fileNames = cms.untracked.vstring('file:single_neutrino.random.dat')
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',                                                       # drop all objects
        'keep MaterialAccountingTracks_trackingMaterialProducer_*_*'),  # but the material accounting informations
    fileName = cms.untracked.string('file:material.root')
)

process.path = cms.Path(process.VtxSmeared*process.trackingMaterialProducer)
process.outpath = cms.EndPath(process.out)
