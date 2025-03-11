import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.track_selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *

import Validation.RecoMuon.MuonTrackValidator_cfi
MTVhlt = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone(
# DEFAULTS ###################################
#    label_tp = "mix:MergedTrackTruth",
#    label_tp_refvector = False,
#    muonTPSelector = dict(muonTPSet),
##############################################
label_tp = ("TPmu"),
label_tp_refvector = True,
dirName = 'HLT/Muon/MuonTrack/',
#beamSpot = 'hltOfflineBeamSpot',
ignoremissingtrackcollection=True,
doSummaryPlots = True
)
MTVhlt.muonTPSelector.src = ("TPmu")
################################################

#
# The HLT Muon Multi Track Validator
#

hltMuonMultiTrackValidator = MTVhlt.clone(
    associatormap = (
        'tpToL2MuonAssociation',
        'tpToL2UpdMuonAssociation',
        'tpToL3OITkMuonAssociation',
        'tpToL3TkMuonAssociation',
        'tpToL3FromL1TkMuonAssociation',
        'tpToL0L3FromL1TkMuonAssociation',
        'tpToL3GlbMuonAssociation',
        'tpToL3NoIDMuonAssociation',
        'tpToL3MuonAssociation'
    ),
    label = (
        'hltL2Muons',
        'hltL2Muons:UpdatedAtVtx',
        'hltIterL3OIMuonTrackSelectionHighPurity:',
        'hltIterL3MuonMerged:',
        'hltIterL3MuonAndMuonFromL1Merged:',
        'hltIter0IterL3FromL1MuonTrackSelectionHighPurity:',
        'hltIterL3GlbMuon:',
        'hltIterL3MuonsNoIDTracks:',
        'hltIterL3MuonsTracks:'
    ),
    muonHistoParameters = (
        staMuonHistoParameters,
        staUpdMuonHistoParameters,
        trkMuonHistoParameters,
        trkMuonHistoParameters,
        trkMuonHistoParameters,
        trkMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters
    )
)

#
# The Phase-2 validator
#

_hltMuonMultiTrackValidator = MTVhlt.clone(
    associatormap = (
        'Phase2tpToL2SeedAssociation',
        'Phase2tpToL2MuonAssociation',
        'Phase2tpToL2MuonUpdAssociation',
        'Phase2tpToL3IOTkAssociation',
        'Phase2tpToL3OITkAssociation',
        'Phase2tpToL3TkMergedAssociation',
        'Phase2tpToL3GlbMuonMergedAssociation',
        'Phase2tpToL3MuonNoIdAssociation',
        'Phase2tpToL3MuonIdAssociation'
    ),
    label = (
        'hltPhase2L2MuonSeedTracks',
        'hltL2MuonsFromL1TkMuon',
        'hltL2MuonsFromL1TkMuon:UpdatedAtVtx',
        'hltIter2Phase2L3FromL1TkMuonMerged',
        'hltPhase2L3OIMuonTrackSelectionHighPurity',
        'hltPhase2L3MuonMerged',
        'hltPhase2L3GlbMuon',
        'hltPhase2L3MuonNoIdTracks',
        'hltPhase2L3MuonIdTracks'
    ),
    muonHistoParameters = (
        staSeedMuonHistoParameters,
        staMuonHistoParameters,
        staUpdMuonHistoParameters,
        trkMuonHistoParameters,
        trkMuonHistoParameters,
        trkMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters
    )
)

# Customization for Inside-Out / Outside-In first approaches
from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons
from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst

def _modify_for_IO_first(validator):
    validator.associatormap += ['Phase2tpToL2MuonToReuseAssociation', 'Phase2tpToL3IOTkFilteredAssociation']
    validator.label += ['hltPhase2L3MuonFilter:L2MuToReuse', 'hltPhase2L3MuonFilter:L3IOTracksFiltered']
    validator.muonHistoParameters.extend([staMuonHistoParameters, trkMuonHistoParameters])

def _modify_for_OI_first(validator):
    validator.associatormap += ['Phase2tpToL3OITkFilteredAssociation']
    validator.label += ['hltPhase2L3MuonFilter:L3OITracksFiltered']
    validator.muonHistoParameters.extend([trkMuonHistoParameters])
    
(phase2L2AndL3Muons & ~phase2L3MuonsOIFirst).toModify(_hltMuonMultiTrackValidator, _modify_for_IO_first)
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toModify(_hltMuonMultiTrackValidator, _modify_for_OI_first)

#
# The full Muon HLT validation sequence
#
muonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq + hltMuonMultiTrackValidator)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(hltMuonMultiTrackValidator, _hltMuonMultiTrackValidator)

recoMuonValidationHLT_seq = cms.Sequence(
    cms.SequencePlaceholder("TPmu") +
    muonValidationHLT_seq
)
