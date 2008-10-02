import FWCore.ParameterSet.Config as cms

def tqafLayer2EventContent(process):
    #
    # define event content on top of layer0 & 1
    #
    process.tqafLayer2EventContent_ttSemiLeptonic = cms.PSet(
        outputCommands = cms.untracked.vstring('keep *_ttSemiLepEvent_*_*',
                                               'keep *_solutions_*_*'
                                               )
    )
    process.tqafEventContent.outputCommands.extend(process.tqafLayer2EventContent_ttSemiLeptonic.outputCommands)
    
    return()
