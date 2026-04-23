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
        'hltIterL3OIMuonTrackSelectionHighPurity',
        'hltIterL3MuonMerged',
        'hltIterL3MuonAndMuonFromL1Merged',
        'hltIter0IterL3FromL1MuonTrackSelectionHighPurity',
        'hltIterL3GlbMuon',
        'hltIterL3MuonsNoIDTracks',
        'hltIterL3MuonsTracks'
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
        'Phase2tpToL3Iter0TkAssociation',
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
        'hltIter0Phase2L3FromL1TkMuonTrackSelectionHighPurity',
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
        trkMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters
    )
)

def _insert_next_to_label(
    validator,
    target_label,
    new_label,
    new_associator,
    new_histo_params,
    insert_after=False,
):
    """Insert a label, its associator and its histogram parameters next to a target label"""
    try:
        idx = list(validator.label).index(target_label)
        insert_idx = idx + 1 if insert_after else idx

        validator.label = (
            list(validator.label[:insert_idx])
            + [new_label]
            + list(validator.label[insert_idx:])
        )
        validator.associatormap = (
            list(validator.associatormap[:insert_idx])
            + [new_associator]
            + list(validator.associatormap[insert_idx:])
        )
        validator.muonHistoParameters = (
            list(validator.muonHistoParameters[:insert_idx])
            + [new_histo_params]
            + list(validator.muonHistoParameters[insert_idx:])
        )
    except ValueError:
        raise RuntimeError(
            f"Target label '{target_label}' not found in validator. "
            f"Available labels: {list(validator.label)}"
        )

def _modify_for_IO_first(validator):
    _insert_next_to_label(
        validator,
        "hltPhase2L3OIMuonTrackSelectionHighPurity",
        "hltPhase2L3MuonFilter:L3IOTracksFiltered",
        "Phase2tpToL3IOTkFilteredAssociation",
        trkMuonHistoParameters,
    )
    _insert_next_to_label(
        validator,
        "hltPhase2L3MuonFilter:L3IOTracksFiltered",
        "hltPhase2L3MuonFilter:L2MuToReuse",
        "Phase2tpToL2MuonToReuseAssociation",
        staMuonHistoParameters,
        insert_after=True,
    )

def _modify_for_OI_first(validator):
    _insert_next_to_label(
        validator,
        "hltPhase2L3OIMuonTrackSelectionHighPurity",
        "hltPhase2L3MuonFilter:L3OITracksFiltered",
        "Phase2tpToL3OITkFilteredAssociation",
        trkMuonHistoParameters,
        insert_after=True,
    )

# Customization for Inside-Out / Outside-In first approaches
from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
(~phase2L3MuonsOIFirst).toModify(_hltMuonMultiTrackValidator, _modify_for_IO_first)
phase2L3MuonsOIFirst.toModify(_hltMuonMultiTrackValidator, _modify_for_OI_first)

def _replace_associator_label(validator, target_label, new_label, new_associator, new_histo_params):
    # Find the index of the target track collection
    idx = list(validator.label).index(target_label)

    validator.associatormap[idx] = new_associator
    validator.label[idx] = new_label
    validator.muonHistoParameters[idx] = new_histo_params

def _modify_for_pixelTracksSelector(validator):
    # Replace the iter0 tracks with the muon pixel tracks before the high-purity selection
    _replace_associator_label(validator,
                              target_label = 'hltIter0Phase2L3FromL1TkMuonTrackSelectionHighPurity',
                              new_label = 'hltPhase2MuonPixelTracks',
                              new_associator = 'Phase2tpToMuonPixelTracksAssociation',
                              new_histo_params = trkMuonHistoParameters
    )

    # Replace the L3 IO tracks with the IO tracks of the pixel-track chain
    _replace_associator_label(validator,
                             target_label = 'hltIter2Phase2L3FromL1TkMuonMerged',
                             new_label = 'hltPhase2MuonIOTracks',
                             new_associator = 'Phase2tpToL3IOTkAssociation',
                             new_histo_params = trkMuonHistoParameters
    )
from Configuration.ProcessModifiers.phase2MuonPixelTracksSelector_cff import phase2MuonPixelTracksSelector
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2MuonPixelTracksSelector | ngtScouting).toModify(_hltMuonMultiTrackValidator, _modify_for_pixelTracksSelector)
# Check that the associators and labels are consistent
# All MTV clones are DQMEDAnalyzers
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
# Access all the global variables
global_items = list(globals().items())
for _name, _obj in global_items:
    # Find all MTV clones
    if isinstance(_obj, DQMEDAnalyzer) and hasattr(_obj, 'label') and hasattr(_obj, 'associatormap') and hasattr(_obj, 'muonHistoParameters'):
        # Check that the size of the associators, lables and muonHistoParameters are the same
        if (len(_obj.label) != len(_obj.associatormap) or len(_obj.label) != len(_obj.muonHistoParameters)
            or len(_obj.associatormap) != len(_obj.muonHistoParameters)):
            raise RuntimeError(f"MuonTrackValidatorHLT -- {_name}: associatormap, label and muonHistoParameters must have the same length!")
        # Check that the trackCollection used in each associator corresponds to the validator's label
        for i in range(0, len(_obj.label)):
            # Dynamically import the associators module to have access to procModifiers changes
            associators_module = __import__('Validation.RecoMuon.associators_cff', globals(), locals(), ['associators'], 0)
            _assoc = getattr(associators_module, _obj.associatormap[i].value()) if isinstance(_obj.associatormap[i], cms.InputTag) else getattr(associators_module, _obj.associatormap[i])
            _label = _obj.label[i].value() if isinstance(_obj.label[i], cms.InputTag) else _obj.label[i]
            _tracksTag = _assoc.tracksTag.value() if hasattr(_assoc, 'tracksTag') else _assoc.label_tr.value()
            if _tracksTag != _label:
                raise RuntimeError(f"MuonTrackValidatorHLT -- {_name}: associatormap and label do not match for index {i}.\n"
                                   f"Associator's tracksTag: {_tracksTag}, collection label in the validator: {_label}.\n"
                                   "Make sure to have the correct ordering!")

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(hltMuonMultiTrackValidator, _hltMuonMultiTrackValidator)

#
# The full Muon HLT validation sequence
#

muonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq + hltMuonMultiTrackValidator)

recoMuonValidationHLT_seq = cms.Sequence(
    cms.SequencePlaceholder("TPmu") +
    muonValidationHLT_seq
)
