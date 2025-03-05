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
ignoremissingtrackcollection=True
)
MTVhlt.muonTPSelector.src = ("TPmu")
################################################

l2MuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL2MuonAssociation',),
    label = ('hltL2Muons',),
    muonHistoParameters = staMuonHistoParameters
)
l2UpdMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL2UpdMuonAssociation',),
    label = ('hltL2Muons:UpdatedAtVtx',),
    muonHistoParameters = staUpdMuonHistoParameters
)
l3OITkMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL3OITkMuonAssociation',),
    label = ('hltIterL3OIMuonTrackSelectionHighPurity:',),
    muonHistoParameters = trkMuonHistoParameters
)
l3TkMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL3TkMuonAssociation',),
    label = ('hltIterL3MuonMerged:',),
    muonHistoParameters = trkMuonHistoParameters
)
l3IOFromL1TkMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL3FromL1TkMuonAssociation',),
    label = ('hltIterL3MuonAndMuonFromL1Merged:',),
    muonHistoParameters = trkMuonHistoParameters
)
l0l3FromL1TkMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL0L3FromL1TkMuonAssociation',),
    label = ('hltIter0IterL3FromL1MuonTrackSelectionHighPurity:',),
    muonHistoParameters = trkMuonHistoParameters
)
l3GlbMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL3GlbMuonAssociation',),
    label = ('hltIterL3GlbMuon:',),
    muonHistoParameters = glbMuonHistoParameters
)
l3NoIDMuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL3NoIDMuonAssociation',),
    label = ('hltIterL3MuonsNoIDTracks:',),
    muonHistoParameters = glbMuonHistoParameters
)
l3MuonMuTrackV = MTVhlt.clone(
    associatormap = ('tpToL3MuonAssociation',),
    label = ('hltIterL3MuonsTracks:',),
    muonHistoParameters = glbMuonHistoParameters
)

#
# The Phase-2 validators
#

# L2 standalone muons seeds
Phase2l2MuSeedV = MTVhlt.clone(
    associatormap = ('Phase2tpToL2SeedAssociation',),
    label = ('hltPhase2L2MuonSeedTracks',),
    muonHistoParameters = (staSeedMuonHistoParameters,)
)
# L2 standalone muons
Phase2l2MuV = MTVhlt.clone(
    associatormap = ('Phase2tpToL2MuonAssociation',),
    label = ('hltL2MuonsFromL1TkMuon',),
    muonHistoParameters = (staMuonHistoParameters,)
)
# L2 standalone muons updated at vertex
Phase2l2MuUpdV = MTVhlt.clone(
    associatormap = ('Phase2tpToL2MuonUpdAssociation',),
    label = ('hltL2MuonsFromL1TkMuon:UpdatedAtVtx',),
    muonHistoParameters = (staUpdMuonHistoParameters,)
)
# L3 IO inner tracks
Phase2l3IOTkV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3IOTkAssociation',),
    label = ('hltIter2Phase2L3FromL1TkMuonMerged',),
    muonHistoParameters = (trkMuonHistoParameters,)
)
# L3 OI inner tracks
Phase2l3OITkV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3OITkAssociation',),
    label = ('hltPhase2L3OIMuonTrackSelectionHighPurity',),
    muonHistoParameters = (trkMuonHistoParameters,)
)
# L2 muons to reuse (IO first only)
Phase2L2MuToReuseV = MTVhlt.clone(
    associatormap = ('Phase2tpToL2MuonToReuseAssociation',),
    label = ('hltPhase2L3MuonFilter:L2MuToReuse',),
    muonHistoParameters = (staMuonHistoParameters,)
)
# L3 IO inner tracks filtered (IO first only)
Phase2l3IOTkFilteredV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3IOTkFilteredAssociation',),
    label = ('hltPhase2L3MuonFilter:L3IOTracksFiltered',),
    muonHistoParameters = (trkMuonHistoParameters,)
)
# L3 OI inner tracks filtered (OI first only)
Phase2l3OITkFilteredV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3OITkFilteredAssociation',),
    label = ('hltPhase2L3MuonFilter:L3OITracksFiltered',),
    muonHistoParameters = (trkMuonHistoParameters,)
)
# L3 inner tracks merged
Phase2l3TkMergedV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3TkMergedAssociation',),
    label = ('hltPhase2L3MuonMerged',),
    muonHistoParameters = (trkMuonHistoParameters,)
)
# L3 global muons
Phase2l3GlbMuonV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3GlbMuonMergedAssociation',),
    label = ('hltPhase2L3GlbMuon',),
    muonHistoParameters = (glbMuonHistoParameters,)
)
# L3 Muons no ID
Phase2l3MuNoIdTrackV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3MuonNoIdAssociation',),
    label = ('hltPhase2L3MuonNoIdTracks',),
    muonHistoParameters = (glbMuonHistoParameters,)
)
# L3 Muons ID
Phase2l3MuIdTrackV = MTVhlt.clone(
    associatormap = ('Phase2tpToL3MuonIdAssociation',),
    label = ('hltPhase2L3MuonIdTracks',),
    muonHistoParameters = (glbMuonHistoParameters,)
)

phase2HltMuonV = MTVhlt.clone(
    associatormap = ('Phase2tpToL2SeedAssociation',
                     'Phase2tpToL2MuonAssociation',
                     'Phase2tpToL2MuonUpdAssociation',
                     'Phase2tpToL3IOTkAssociation',
                     'Phase2tpToL3OITkAssociation',
                     'Phase2tpToL3TkMergedAssociation',
                     'Phase2tpToL3GlbMuonMergedAssociation',
                     'Phase2tpToL3MuonNoIdAssociation',
                     'Phase2tpToL3MuonIdAssociation'),
    label = ('hltPhase2L2MuonSeedTracks',
             'hltL2MuonsFromL1TkMuon',
             'hltL2MuonsFromL1TkMuon:UpdatedAtVtx',
             'hltIter2Phase2L3FromL1TkMuonMerged',
             'hltPhase2L3OIMuonTrackSelectionHighPurity',
             'hltPhase2L3MuonMerged',
             'hltPhase2L3GlbMuon',
             'hltPhase2L3MuonNoIdTracks',
             'hltPhase2L3MuonIdTracks'),
    muonHistoParameters = (staSeedMuonHistoParameters,
                           staMuonHistoParameters,
                           staUpdMuonHistoParameters,
                           trkMuonHistoParameters,
                           trkMuonHistoParameters,
                           trkMuonHistoParameters,
                           glbMuonHistoParameters,
                           glbMuonHistoParameters,
                           glbMuonHistoParameters)
)

#
# The full Muon HLT validation sequence
#

muonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq
                                    +l2MuonMuTrackV
                                    +l2UpdMuonMuTrackV
                                    +l3OITkMuonMuTrackV
                                    +l3TkMuonMuTrackV
                                    +l3IOFromL1TkMuonMuTrackV 
                                    +l0l3FromL1TkMuonMuTrackV
                                    +l3GlbMuonMuTrackV
                                    +l3NoIDMuonMuTrackV
                                    +l3MuonMuTrackV
                                    )

#
# The Phase-2 sequences
#

Phase2MuonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq + phase2HltMuonV)

_Phase2MuonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq
                                    +Phase2l2MuSeedV
                                    +Phase2l2MuV
                                    +Phase2l2MuUpdV
                                    +Phase2l3IOTkV
                                    +Phase2l3OITkV
                                    +Phase2l3TkMergedV
                                    +Phase2l3GlbMuonV
                                    +Phase2l3MuNoIdTrackV
                                    +Phase2l3MuIdTrackV
                                    )
# Inside-Out first sequence
Phase2IOFirstMuonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq
                                    +Phase2l2MuSeedV
                                    +Phase2l2MuV
                                    +Phase2l2MuUpdV
                                    +Phase2l3IOTkV
                                    +Phase2L2MuToReuseV
                                    +Phase2l3IOTkFilteredV
                                    +Phase2l3OITkV
                                    +Phase2l3TkMergedV
                                    +Phase2l3GlbMuonV
                                    +Phase2l3MuNoIdTrackV
                                    +Phase2l3MuIdTrackV
                                    )
# Outside-In first sequence
Phase2OIFirstMuonValidationHLT_seq = cms.Sequence(muonAssociationHLT_seq
                                    +Phase2l2MuSeedV
                                    +Phase2l2MuV
                                    +Phase2l2MuUpdV
                                    +Phase2l3OITkV
                                    +Phase2l3OITkFilteredV
                                    +Phase2l3IOTkV
                                    +Phase2l3TkMergedV
                                    +Phase2l3GlbMuonV
                                    +Phase2l3MuNoIdTrackV
                                    +Phase2l3MuIdTrackV
                                    )

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(muonValidationHLT_seq, Phase2MuonValidationHLT_seq)

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons
phase2L2AndL3Muons.toReplaceWith(muonValidationHLT_seq, Phase2IOFirstMuonValidationHLT_seq)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toReplaceWith(muonValidationHLT_seq, Phase2OIFirstMuonValidationHLT_seq)

recoMuonValidationHLT_seq = cms.Sequence(
    cms.SequencePlaceholder("TPmu") +
    muonValidationHLT_seq
    )
