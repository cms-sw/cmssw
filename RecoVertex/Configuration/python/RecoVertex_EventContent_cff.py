import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing

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

_phase2_tktiming_RecoVertexEventContent = [ 'keep *_offlinePrimaryVertices1D__*',
                                            'keep *_offlinePrimaryVerticesLegacy__*',
                                            'keep *_offlinePrimaryVertices1DWithBS__*',
                                            'keep *_offlinePrimaryVerticesLegacyWithBS__*',
                                            'keep *_trackTimeValueMapProducer_*_*' ]

def _phase2_tktiming_AddNewContent(mod):
    temp = mod.outputCommands + _phase2_tktiming_RecoVertexEventContent
    phase2_timing.toModify( mod, outputCommands = temp )

_phase2_tktiming_AddNewContent(RecoVertexFEVT)
_phase2_tktiming_AddNewContent(RecoVertexRECO)
_phase2_tktiming_AddNewContent(RecoVertexAOD)


