import FWCore.ParameterSet.Config as cms

#
# define event content of common tqafLayer1
#
def makeTqafLayer1EventContent(process):

    ## define event content
    process.tqafEventContent = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *')
    )

    ## std pat layer1 event content
    process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
    process.tqafEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)
    
    ## additionally reco'ed pat layer1 event content
    process.tqafEventContent_pat = cms.PSet(
        outputCommands = cms.untracked.vstring('keep *_selectedLayer1CaloTaus_*_*' ## reco'd caloTaus
                                               )
    )
    process.tqafEventContent.outputCommands.extend(process.tqafEventContent_pat.outputCommands)
        
    ## additional reco/aod event content
    process.tqafEventContent_aod = cms.PSet(
        outputCommands = cms.untracked.vstring('keep *_genParticles_*_*',          ## all genPaticles (unpruned)
                                               'keep *_genEventScale_*_*',         ## genEvent info
                                               'keep *_genEventWeight_*_*',        ## genEvent info
                                               'keep *_genEventProcID_*_*',        ## genEvent info
                                               'keep *_genEventPdfInfo_*_*',       ## genEvent info
                                               'keep *_genEventRunInfo_*_*',       ## genEvent info
                                               'keep *_hltTriggerSummaryAOD_*_*',  ## hlt TriggerEvent
                                               'keep *_towerMaker_*_*',            ## all caloTowers
                                               'keep *_offlineBeamSpot_*_*'        ## offline beamspot
                                               )
    )
    process.tqafEventContent.outputCommands.extend(process.tqafEventContent_aod.outputCommands)

    return()
