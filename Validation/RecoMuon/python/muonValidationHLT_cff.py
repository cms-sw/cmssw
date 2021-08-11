import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.track_selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *

import Validation.RecoMuon.MuonTrackValidator_cfi
MTVhlt = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
# DEFAULTS ###################################
#    label_tp = cms.InputTag("mix","MergedTrackTruth"),
#    label_tp_refvector = cms.bool(False),
#    muonTPSelector = cms.PSet(muonTPSet),
##############################################
MTVhlt.label_tp = ("TPmu")
MTVhlt.label_tp_refvector = True
MTVhlt.muonTPSelector.src = ("TPmu")
MTVhlt.dirName = 'HLT/Muon/MuonTrack/'
#MTVhlt.beamSpot = 'hltOfflineBeamSpot'
MTVhlt.ignoremissingtrackcollection=True
################################################

l2MuonMuTrackV = MTVhlt.clone()
l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociation'
l2MuonMuTrackV.label = ('hltL2Muons',)
l2MuonMuTrackV.muonHistoParameters = staMuonHistoParameters

l2UpdMuonMuTrackV = MTVhlt.clone()
l2UpdMuonMuTrackV.associatormap = 'tpToL2UpdMuonAssociation'
l2UpdMuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonMuTrackV.muonHistoParameters = staUpdMuonHistoParameters

l3OITkMuonMuTrackV = MTVhlt.clone()
l3OITkMuonMuTrackV.associatormap = 'tpToL3OITkMuonAssociation'
l3OITkMuonMuTrackV.label = ('hltIterL3OIMuonTrackSelectionHighPurity:',)
l3OITkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3TkMuonMuTrackV = MTVhlt.clone()
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociation'
l3TkMuonMuTrackV.label = ('hltIterL3MuonMerged:',)
l3TkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3IOFromL1TkMuonMuTrackV = MTVhlt.clone()
l3IOFromL1TkMuonMuTrackV.associatormap = 'tpToL3FromL1TkMuonAssociation'
l3IOFromL1TkMuonMuTrackV.label = ('hltIterL3MuonAndMuonFromL1Merged:',)
l3IOFromL1TkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3GlbMuonMuTrackV = MTVhlt.clone()
l3GlbMuonMuTrackV.associatormap = 'tpToL3GlbMuonAssociation'
l3GlbMuonMuTrackV.label = ('hltIterL3GlbMuon:',)
l3GlbMuonMuTrackV.muonHistoParameters = glbMuonHistoParameters

l3NoIDMuonMuTrackV = MTVhlt.clone()
l3NoIDMuonMuTrackV.associatormap = 'tpToL3NoIDMuonAssociation'
l3NoIDMuonMuTrackV.label = ('hltIterL3MuonsNoIDTracks:',)
l3NoIDMuonMuTrackV.muonHistoParameters = glbMuonHistoParameters

l3MuonMuTrackV = MTVhlt.clone()
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociation'
l3MuonMuTrackV.label = ('hltIterL3MuonsTracks:',)
l3MuonMuTrackV.muonHistoParameters = glbMuonHistoParameters

#
# The full Muon HLT validation sequence
#
muonValidationHLT_seq = cms.Sequence(
    tpToL2MuonAssociation + l2MuonMuTrackV
    +tpToL2UpdMuonAssociation + l2UpdMuonMuTrackV
    +tpToL3OITkMuonAssociation + l3OITkMuonMuTrackV
    +tpToL3TkMuonAssociation + l3TkMuonMuTrackV
    +tpToL3FromL1TkMuonAssociation + l3IOFromL1TkMuonMuTrackV
    +tpToL3GlbMuonAssociation + l3GlbMuonMuTrackV
    +hltIterL3MuonsNoIDTracks_seq + tpToL3NoIDMuonAssociation + l3NoIDMuonMuTrackV
    +hltIterL3MuonsTracks_seq + tpToL3MuonAssociation + l3MuonMuTrackV
    )

recoMuonValidationHLT_seq = cms.Sequence(
    cms.SequencePlaceholder("TPmu") +
    muonValidationHLT_seq
    )
