import FWCore.ParameterSet.Config as cms

tqafEventContent = [
  ## genEvt
  'keep *_decaySubset_*_*',
  'keep *_initSubset_*_*', 
  'keep *_genEvt_*_*',
  ## extras for event selection
  'keep *_kinFitTtSemiLepEventSelection_*_*',
  'keep *_findTtSemiLepSignalSelMVA_*_*',
  ## hypotheses & event structure
  'keep *_ttSemiLepHyp*_*_*',
  'keep *_ttSemiLepEvent_*_*',
  'keep *_ttFullLepHyp*_*_*',
  'keep *_ttFullLepEvent_*_*',  
  'keep *_ttFullHadHyp*_*_*',
  'keep *_ttFullHadEvent_*_*'
]
