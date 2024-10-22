import FWCore.ParameterSet.Config as cms

#AOD content
RecoVertexAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices__*', 
        'keep *_offlinePrimaryVerticesWithBS_*_*',
        'keep *_offlinePrimaryVerticesFromCosmicTracks_*_*',
        'keep *_nuclearInteractionMaker_*_*',
        'keep *_generalV0Candidates_*_*',                                           
	'keep *_inclusiveSecondaryVertices_*_*')
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer

_phase2_tktiming_RecoVertexEventContent = [ 'keep *_offlinePrimaryVertices4D__*',
                                            'keep *_offlinePrimaryVertices4DWithBS__*',
                                            'keep *_trackTimeValueMapProducer_*_*' ]

_phase2_tktiming_layer_RecoVertexEventContent = [ 'keep *_tofPID_*_*']
phase2_timing.toModify( RecoVertexAOD,
     outputCommands = RecoVertexAOD.outputCommands + _phase2_tktiming_RecoVertexEventContent)
phase2_timing_layer.toModify( RecoVertexAOD,
     outputCommands = RecoVertexAOD.outputCommands + _phase2_tktiming_layer_RecoVertexEventContent)

#RECO content
RecoVertexRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoVertexRECO.outputCommands.extend(RecoVertexAOD.outputCommands)

#FEVT content
RecoVertexFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoVertexFEVT.outputCommands.extend(RecoVertexRECO.outputCommands)
