import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.TtFullHadKinFitter = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.KinFitter = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary      = cms.untracked.bool(True)
)

## configure geometry & conditions
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.task = cms.Task()

## std sequence for pat
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.task.add(process.patCandidatesTask)
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)

## do event filtering on generator level
process.load("TopQuarkAnalysis.TopSkimming.ttDecayChannelFilters_cff")
process.task.add(process.ttDecayChannelFiltersTask)

## std sequence to produce the kinematic fit for full hadronic events
process.load("TopQuarkAnalysis.TopKinFitter.TtFullHadKinFitProducer_cfi")
process.task.add(process.kinFitTtFullHadEvent)

## process path
process.p = cms.Path(
    process.ttFullHadronicFilter,
    process.task
)

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),
    fileName = cms.untracked.string('ttFullHadKinFitProducer.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_kinFitTtFullHadEvent_*_*']

## output path
process.outpath = cms.EndPath(process.out)
