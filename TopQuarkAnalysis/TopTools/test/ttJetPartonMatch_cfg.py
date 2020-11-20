import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.JetPartonMatching = cms.untracked.PSet(
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
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')
process.load("Configuration.StandardSequences.MagneticField_cff")

process.task = cms.Task()

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.task.add(process.patCandidatesTask)
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
process.task.add(process.makeGenEvtTask)

## configure jet-parton matching
process.load("TopQuarkAnalysis.TopTools.TtFullHadJetPartonMatch_cfi")
process.task.add(process.ttFullHadJetPartonMatch)
process.ttFullHadJetPartonMatch.verbosity  = 1              #default: 0
#process.ttFullHadJetPartonMatch.algorithm  = "minSumDist"   #default: totalMinDist
#process.ttFullHadJetPartonMatch.useDeltaR  = True           #default: True
#process.ttFullHadJetPartonMatch.useMaxDist = True           #default: False
#process.ttFullHadJetPartonMatch.maxDist    = 2.5            #default: 0.3
#process.ttFullHadJetPartonMatch.maxNJets   = 7              #default: 6
#process.ttFullHadJetPartonMatch.maxNComb   = 1              #default: 1
process.load("TopQuarkAnalysis.TopTools.TtFullLepJetPartonMatch_cfi")
process.task.add(process.ttFullLepJetPartonMatch)
process.ttFullLepJetPartonMatch.verbosity  = 1              #default: 0
#process.ttFullLepJetPartonMatch.algorithm  = "minSumDist"   #default: totalMinDist
#process.ttFullLepJetPartonMatch.useDeltaR  = True           #default: True
#process.ttFullLepJetPartonMatch.useMaxDist = True           #default: False
#process.ttFullLepJetPartonMatch.maxDist    = 2.5            #default: 0.3
#process.ttFullLepJetPartonMatch.maxNJets   = 3              #default: 2
#process.ttFullLepJetPartonMatch.maxNComb   = 1              #default: 1
process.load("TopQuarkAnalysis.TopTools.TtSemiLepJetPartonMatch_cfi")
process.task.add(process.ttSemiLepJetPartonMatch)
process.ttSemiLepJetPartonMatch.verbosity  = 1              #default: 0
#process.ttSemiLepJetPartonMatch.algorithm  = "minSumDist"   #default: totalMinDist
#process.ttSemiLepJetPartonMatch.useDeltaR  = True           #default: True
#process.ttSemiLepJetPartonMatch.useMaxDist = True           #default: False
#process.ttSemiLepJetPartonMatch.maxDist    = 2.5            #default: 0.3
#process.ttSemiLepJetPartonMatch.maxNJets   = 5              #default: 4
#process.ttSemiLepJetPartonMatch.maxNComb   = 1              #default: 1

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ttJetPartonMatch.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_ttFullHadJetPartonMatch_*_*',
                               'keep *_ttFullLepJetPartonMatch_*_*',
                               'keep *_ttSemiLepJetPartonMatch_*_*']

## output path
process.outpath = cms.EndPath(process.out, process.task)
