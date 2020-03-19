import FWCore.ParameterSet.Config as cms

RecoVertexFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices__*', 
        'keep *_offlinePrimaryVerticesWithBS_*_*',
        'keep *_offlinePrimaryVerticesFromCosmicTracks_*_*',
        'keep *_nuclearInteractionMaker_*_*',
        'keep *_generalV0Candidates_*_*',
	'keep *_inclusiveSecondaryVertices_*_*')
)
#RECO content
RecoVertexRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices__*', 
        'keep *_offlinePrimaryVerticesWithBS_*_*',
        'keep *_offlinePrimaryVerticesFromCosmicTracks_*_*',
        'keep *_nuclearInteractionMaker_*_*',
        'keep *_generalV0Candidates_*_*',
	'keep *_inclusiveSecondaryVertices_*_*')
)
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

_phase2_tktiming_layer_RecoVertexEventContent = [ 'keep *_offlinePrimaryVertices4DnoPID__*',
                                            'keep *_offlinePrimaryVertices4DnoPIDWithBS__*',
                                            'keep *_tofPID_*_*']

def _phase2_tktiming_AddNewContent(mod):
    temp = mod.outputCommands + _phase2_tktiming_RecoVertexEventContent
    phase2_timing.toModify( mod, outputCommands = temp )

_phase2_tktiming_AddNewContent(RecoVertexFEVT)
_phase2_tktiming_AddNewContent(RecoVertexRECO)
_phase2_tktiming_AddNewContent(RecoVertexAOD)

def _phase2_tktiming_layer_AddNewContent(mod):
    temp = mod.outputCommands + _phase2_tktiming_layer_RecoVertexEventContent
    phase2_timing_layer.toModify( mod, outputCommands = temp )

_phase2_tktiming_layer_AddNewContent(RecoVertexFEVT)
_phase2_tktiming_layer_AddNewContent(RecoVertexRECO)
_phase2_tktiming_layer_AddNewContent(RecoVertexAOD)
