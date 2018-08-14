import FWCore.ParameterSet.Config as cms

rootfile="out.root"

calojetcoll="iterativeConePu5CaloJets"
#calojetcoll="hltIterativeCone5PileupSubtractionCaloJets"
genjetcoll="iterativeCone5HiGenJets"

hltlow35  =""
hltname35="HLT_HIJet35U"
folderjet35="HLT/HLTJETMET/SingleJet35U"

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SingleJetPathVal35 = DQMEDAnalyzer('HLTJetMETValidation',
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    DQMFolder             = cms.untracked.string(folderjet35),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
    HLTLow                = cms.untracked.InputTag(hltlow35),
    HLTPath               = cms.untracked.InputTag(hltname35),
    PFJetAlgorithm        = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag(genjetcoll),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetCalo"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
)

hiSingleJetValidation = cms.Sequence(SingleJetPathVal35)
