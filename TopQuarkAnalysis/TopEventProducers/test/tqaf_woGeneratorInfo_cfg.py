import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

## define input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesSingleMuRECO
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( filesSingleMuRECO )
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

## configure geometry & conditions
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:com10')

#-------------------------------------------------
# PAT and TQAF configuration
#-------------------------------------------------

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.patSequences_cff")

process.patJets.addTagInfos = False

## std sequence for TQAF
process.load("TopQuarkAnalysis.TopEventProducers.tqafSequences_cff")

## remove MC specific stuff in TQAF
process.tqafTtSemiLeptonic.remove(process.makeGenEvt)
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff import *
addTtSemiLepHypotheses(process, ["kGeom", "kWMassMaxSumPt", "kMaxSumPtWMass"])
removeTtSemiLepHypGenMatch(process)

## process path
process.p = cms.Path(process.patDefaultSequence *
                     process.tqafTtSemiLeptonic
                     )

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName       = cms.untracked.string('tqafOutput.woGeneratorInfo.root'),
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *'),
    dropMetaData   = cms.untracked.string("DROPPED")  ## NONE    for none
                                                      ## DROPPED for drop for dropped data
)
process.outpath = cms.EndPath(process.out)

### remove MC specific stuff in PAT
#from PhysicsTools.PatAlgos.tools.coreTools import *
#removeMCMatching(process, ["All"])
# FIXME: very (too) simple to replace functionality from removed coreTools.py
from PhysicsTools.PatAlgos.tools.helpers import removeIfInSequence
process.patElectrons.addGenMatch  = False
removeIfInSequence(process, 'electronMatch', "patDefaultSequence")
process.patJets.addGenPartonMatch = False
removeIfInSequence(process, 'patJetPartons', "patDefaultSequence")
removeIfInSequence(process, 'patJetPartonAssociation', "patDefaultSequence")
removeIfInSequence(process, 'patJetPartonMatch', "patDefaultSequence")
process.patJets.addGenJetMatch    = False
removeIfInSequence(process, 'patJetGenJetMatch', "patDefaultSequence")
process.patJets.getJetMCFlavour   = False
removeIfInSequence(process, 'patJetFlavourId', "patDefaultSequence")
removeIfInSequence(process, 'patJetFlavourAssociation', "patDefaultSequence")
process.patMETs.addGenMET         = False
process.patMuons.addGenMatch      = False
removeIfInSequence(process, 'muonMatch', "patDefaultSequence")
process.patPhotons.addGenMatch    = False
removeIfInSequence(process, 'photonMatch', "patDefaultSequence")
process.patTaus.addGenMatch       = False
removeIfInSequence(process, 'tauMatch', "patDefaultSequence")
process.patTaus.addGenJetMatch    = False
removeIfInSequence(process, 'tauGenJets', "patDefaultSequence")
removeIfInSequence(process, 'tauGenJetsSelectorAllHadrons', "patDefaultSequence")
removeIfInSequence(process, 'tauGenJetMatch', "patDefaultSequence")
process.patJetCorrFactors.levels.append( 'L2L3Residual' )

## PAT content
from PhysicsTools.PatAlgos.patEventContent_cff import *
process.out.outputCommands += patTriggerEventContent
process.out.outputCommands += patExtraAodEventContent
process.out.outputCommands += patEventContentNoCleaning

## TQAF content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import *
process.out.outputCommands += tqafEventContent
