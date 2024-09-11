import FWCore.ParameterSet.Config as cms

#RECO content
RecoPixelVertexingRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*')
)

#Full Event content 
RecoPixelVertexingFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring() 
)
RecoPixelVertexingFEVT.outputCommands.extend(RecoPixelVertexingRECO.outputCommands)
