import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *
import Validation.RecoMuon.MuonTrackValidator_cfi

l3TkMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociation'
l3TkMuonMuTrackV.label = ('hltIterL3MuonMerged:',)
l3TkMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonMuTrackV.ignoremissingtrackcollection=True
l3TkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3OITkMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3OITkMuonMuTrackV.associatormap = 'tpToL3OITkMuonAssociation'
l3OITkMuonMuTrackV.label = ('hltIterL3OIMuonTrackSelectionHighPurity:',)
l3OITkMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#lOI3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3OITkMuonMuTrackV.ignoremissingtrackcollection=True
l3OITkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3IOFromL1TkMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3IOFromL1TkMuonMuTrackV.associatormap = 'tpToL3FromL1TkMuonAssociation'
l3IOFromL1TkMuonMuTrackV.label = ('hltIterL3MuonAndMuonFromL1Merged:',)
l3IOFromL1TkMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#lIOFromL13TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3IOFromL1TkMuonMuTrackV.ignoremissingtrackcollection=True
l3IOFromL1TkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3MuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociation'
l3MuonMuTrackV.label = ('hltIterL3Muons:',)
l3MuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonMuTrackV.ignoremissingtrackcollection=True
l3MuonMuTrackV.muonHistoParameters = glbMuonHistoParameters

l3GlbMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3GlbMuonMuTrackV.associatormap = 'tpToL3GlbMuonAssociation'
l3GlbMuonMuTrackV.label = ('hltIterL3GlbMuon:',)
l3GlbMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#lGlb3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3GlbMuonMuTrackV.ignoremissingtrackcollection=True
l3GlbMuonMuTrackV.muonHistoParameters = glbMuonHistoParameters

l3NoIDMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3NoIDMuonMuTrackV.associatormap = 'tpToL3NoIDMuonAssociation'
l3NoIDMuonMuTrackV.label = ('hltIterL3MuonsNoID:',)
l3NoIDMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#lNoID3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3NoIDMuonMuTrackV.ignoremissingtrackcollection=True
l3NoIDMuonMuTrackV.muonHistoParameters = glbMuonHistoParameters
#
# The full Muon HLT validation sequence
#
muonValidationHLT_seq = cms.Sequence(
    tpToL3TkMuonAssociation + l3TkMuonMuTrackV
    +tpToL3OITkMuonAssociation + l3OITkMuonMuTrackV
    +tpToL3FromL1TkMuonAssociation + l3IOFromL1TkMuonMuTrackV
    +tpToL3MuonAssociation + l3MuonMuTrackV
    +tpToL3GlbMuonAssociation + l3GlbMuonMuTrackV
    +tpToL3NoIDMuonAssociation + l3NoIDMuonMuTrackV
    )

recoMuonValidationHLT_seq = cms.Sequence(
    muonValidationHLT_seq
    )
