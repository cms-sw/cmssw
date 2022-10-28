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
    associatormap = 'tpToL2MuonAssociation',
    label = ('hltL2Muons',),
    muonHistoParameters = staMuonHistoParameters
)
l2UpdMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL2UpdMuonAssociation',
    label = ('hltL2Muons:UpdatedAtVtx',),
    muonHistoParameters = staUpdMuonHistoParameters
)
l3OITkMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL3OITkMuonAssociation',
    label = ('hltIterL3OIMuonTrackSelectionHighPurity:',),
    muonHistoParameters = trkMuonHistoParameters
)
l3TkMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL3TkMuonAssociation',
    label = ('hltIterL3MuonMerged:',),
    muonHistoParameters = trkMuonHistoParameters
)
l3IOFromL1TkMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL3FromL1TkMuonAssociation',
    label = ('hltIterL3MuonAndMuonFromL1Merged:',),
    muonHistoParameters = trkMuonHistoParameters
)
l0l3FromL1TkMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL0L3FromL1TkMuonAssociation',
    label = ('hltIter0IterL3FromL1MuonTrackSelectionHighPurity:',),
    muonHistoParameters = trkMuonHistoParameters
)
l3GlbMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL3GlbMuonAssociation',
    label = ('hltIterL3GlbMuon:',),
    muonHistoParameters = glbMuonHistoParameters
)
l3NoIDMuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL3NoIDMuonAssociation',
    label = ('hltIterL3MuonsNoIDTracks:',),
    muonHistoParameters = glbMuonHistoParameters
)
l3MuonMuTrackV = MTVhlt.clone(
    associatormap = 'tpToL3MuonAssociation',
    label = ('hltIterL3MuonsTracks:',),
    muonHistoParameters = glbMuonHistoParameters
)
#
# The full Muon HLT validation sequence
#
muonValidationHLT_seq = cms.Sequence(
    tpToL2MuonAssociation + l2MuonMuTrackV
    +tpToL2UpdMuonAssociation + l2UpdMuonMuTrackV
    +tpToL3OITkMuonAssociation + l3OITkMuonMuTrackV
    +tpToL3TkMuonAssociation + l3TkMuonMuTrackV
    +tpToL3FromL1TkMuonAssociation + l3IOFromL1TkMuonMuTrackV 
    +tpToL0L3FromL1TkMuonAssociation + l0l3FromL1TkMuonMuTrackV
    +tpToL3GlbMuonAssociation + l3GlbMuonMuTrackV
    +hltIterL3MuonsNoIDTracks_seq + tpToL3NoIDMuonAssociation + l3NoIDMuonMuTrackV
    +hltIterL3MuonsTracks_seq + tpToL3MuonAssociation + l3MuonMuTrackV
    )

recoMuonValidationHLT_seq = cms.Sequence(
    cms.SequencePlaceholder("TPmu") +
    muonValidationHLT_seq
    )
