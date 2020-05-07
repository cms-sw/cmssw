import FWCore.ParameterSet.Config as cms

#AOD content
BeamSpotAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_offlineBeamSpot_*_*')
)

#RECO content
BeamSpotRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
BeamSpotRECO.outputCommands.extend(BeamSpotAOD.outputCommands)

#Full Event content
BeamSpotFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
BeamSpotFEVT.outputCommands.extend(BeamSpotRECO.outputCommands)

