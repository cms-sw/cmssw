import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/CMSSW_3_2_7/Cosmics/RECO/CRAFT09_R_V4_CollisionsSeq-v1/0001/F834D959-F9B1-DE11-8996-001A92810AB2.root'    
    )
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('CRAFT09_R_V8::All')

#-------------------------------------------------
# PAT and TQAF configuration
#-------------------------------------------------

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## switch to sisCone5
from PhysicsTools.PatAlgos.tools.jetTools import *
process.patDefaultSequence.remove(process.ak5JetID)
addJetID(process,
         cms.InputTag('sisCone5CaloJets'),
         'sc5')
switchJetCollection(process, 
                    cms.InputTag('sisCone5CaloJets'),
                    doJTA            = True,           
                    doBTagging       = True,           
                    jetCorrLabel     = ('SC5','Calo'),
                    doType1MET       = True,          
                    genJetCollection = cms.InputTag("sisCone5GenJets"),
                    doJetID          = True,
                    jetIdLabel       = "sc5"
                    ) 

## remove MC specific stuff in PAT
from PhysicsTools.PatAlgos.tools.coreTools import *
removeMCMatching(process, "All")

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

## PAT content
from PhysicsTools.PatAlgos.patEventContent_cff import *
process.out.outputCommands += patTriggerEventContent
process.out.outputCommands += patExtraAodEventContent
process.out.outputCommands += patEventContentNoLayer1Cleaning

## TQAF content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import *
process.out.outputCommands += tqafEventContent
