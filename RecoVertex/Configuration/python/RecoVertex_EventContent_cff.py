import FWCore.ParameterSet.Config as cms

RecoVertexFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*')
)
#RECO content
RecoVertexRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*')
)
#AOD content
RecoVertexAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*')
)

