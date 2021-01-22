import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.TtSemiLeptonicEvent=dict()
process.MessageLogger.TtSemiLepKinFitter=dict()

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#    ['rfio:///castor/cern.ch/user/s/snaumann/test/Spring12/TTJets_TuneZ2star_8TeV-madgraph-tauola_AODSIM_PU_S7_START52_V5-v1/PAT_muJets_1.root',
#     'rfio:///castor/cern.ch/user/s/snaumann/test/Spring12/TTJets_TuneZ2star_8TeV-madgraph-tauola_AODSIM_PU_S7_START52_V5-v1/PAT_muJets_2.root',
#     'rfio:///castor/cern.ch/user/s/snaumann/test/Spring12/TTJets_TuneZ2star_8TeV-madgraph-tauola_AODSIM_PU_S7_START52_V5-v1/PAT_muJets_3.root']
#    )
#)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
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

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff")
process.task.add(process.patCandidatesTask)
#Temporary customize to the unit tests that fail due to old input samples
process.patTaus.skipMissingTauID = True
process.load("PhysicsTools.PatAlgos.selectionLayer1.selectedPatCandidates_cff")
process.task.add(process.selectedPatCandidatesTask)

## use genParticles to select only muon+jets events
process.load("TopQuarkAnalysis.TopSkimming.ttDecayChannelFilters_cff")
process.task.add(process.ttDecayChannelFiltersTask)
process.ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.electron = False

## sequences for ttGenEvent and TtSemiLeptonicEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
process.task.add(process.makeGenEvtTask)

process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff")
process.task.add(process.makeTtSemiLepEventTask)
## enable additional per-event printout from the TtSemiLeptonicEvent
process.ttSemiLepEvent.verbosity = 1

## choose which hypotheses to produce
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff import *
addTtSemiLepHypotheses(process, ["kMaxSumPtWMass", "kKinFit"])
## change some common parameters
#setForAllTtSemiLepHypotheses(process, "maxNJets", 5)
#setForAllTtSemiLepHypotheses(process, "neutrinoSolutionType", 2)
## use b-tagging for hypotheses (neglected for GenMatch)
#setForAllTtSemiLepHypotheses(process, "useBTagging", True)
#setForAllTtSemiLepHypotheses(process, "bTagAlgorithm", "combinedSecondaryVertexBJetTags")
#setForAllTtSemiLepHypotheses(process, "minBDiscBJets"    , 0.679)
#setForAllTtSemiLepHypotheses(process, "maxBDiscLightJets", 0.679)

## change jet-parton matching algorithm
#process.ttSemiLepJetPartonMatch.algorithm = "unambiguousOnly"
## change constraints used in kinematic fit
#process.kinFitTtSemiLepEventHypothesis.constraints = [1,2,6]

## load HypothesisAnalyzer
process.load("TopQuarkAnalysis.Examples.HypothesisAnalyzer_cff")

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopHypothesis.root')
)

## end path
process.path = cms.Path(process.analyzeHypotheses, process.task)
