import FWCore.ParameterSet.Config as cms

#
# tqaf event content (on top of pat layer0&1)
#
tqafLayer2TtSemiLeptonicEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_initSubset_*_*', 
        'keep *_decaySubset_*_*', 
        'keep *_genEvt_*_*',
        'keep *_ttSemiEvent_*_*',
        'keep *_solutions_*_*')
)
