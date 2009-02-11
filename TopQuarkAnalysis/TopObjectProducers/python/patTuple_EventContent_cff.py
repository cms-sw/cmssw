import FWCore.ParameterSet.Config as cms

#
# define event content of common patTuple
#
def makePatTupleEventContent(process):

    ## define event content
    process.patTupleEventContent = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *')
    )
    
    ## std pat layer1 event content
    process.load("PhysicsTools.PatAlgos.patLayer1_EventContent_cff")
    process.patTupleEventContent.outputCommands.extend(process.patLayer1EventContent.outputCommands)
    
    ## additionally reco'ed pat layer1 event content
    process.patTupleEventContent_pat = cms.PSet(
        outputCommands = cms.untracked.vstring('keep *_selectedLayer1CaloTaus_*_*' ## reco'd caloTaus
                                               )
    )
    process.patTupleEventContent.outputCommands.extend(process.patTupleEventContent_pat.outputCommands)
        
    ## additional aod event content
    process.patTupleEventContent_aod = cms.PSet(
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
    process.patTupleEventContent.outputCommands.extend(process.patTupleEventContent_aod.outputCommands)

    return()
