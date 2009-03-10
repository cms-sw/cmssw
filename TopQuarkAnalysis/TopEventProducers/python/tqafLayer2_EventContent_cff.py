import FWCore.ParameterSet.Config as cms

def makeTqafLayer2EventContent(process):
    #
    # define event content on top of layer0 & 1
    #
    process.tqafLayer2EventContent_ttSemiLeptonic = cms.PSet(
        outputCommands = cms.untracked.vstring('keep *_decaySubset_*_*',                   ## genEvent info
                                               'keep *_initSubset_*_*',                    ## genEvent info
                                               'keep *_genEvt_*_*',                        ## genEvent info
                                               'keep *_kinFitTtSemiLepEventSelection_*_*', ## kin fit result for event selection
                                               'keep *_findTtSemiLepSignalSelMVA_*_*',     ## mva result for event selection
                                               'keep *_ttSemiLepEvent_*_*',                ## jet parton association 
                                               'keep *_solutions_*_*'                      ## event solutions (legacy)
                                               )
    )
    process.tqafEventContent.outputCommands.extend(process.tqafLayer2EventContent_ttSemiLeptonic.outputCommands)
    
    return()
