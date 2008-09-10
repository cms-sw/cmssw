import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

import Validation.RecoMuon.RecoMuonValidator_cfi

RecoMuonVMuAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()

RecoMuonVMuAssoc.subDir = 'RecoMuonV/MuonAssoc'
RecoMuonVMuAssoc.simLabel = 'muonTP'

RecoMuonVMuAssoc.trkMuLabel = 'generalTracks'
RecoMuonVMuAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
RecoMuonVMuAssoc.glbMuLabel = 'globalMuons'

RecoMuonVMuAssoc.trkMuAssocLabel = 'tpToTkMuonAssociation'
RecoMuonVMuAssoc.staMuAssocLabel = 'tpToStaMuonAssociation'
RecoMuonVMuAssoc.glbMuAssocLabel = 'tpToGlbMuonAssociation'

RecoMuonVTrackAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()

RecoMuonVTrackAssoc.subDir = 'RecoMuonV/TrackAssoc'
RecoMuonVTrackAssoc.simLabel = 'muonTP'

RecoMuonVTrackAssoc.trkMuLabel = 'generalTracks'
RecoMuonVTrackAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
RecoMuonVTrackAssoc.glbMuLabel = 'globalMuons'

RecoMuonVTrackAssoc.trkMuAssocLabel = 'tpToTkmuTrackAssociation'
RecoMuonVTrackAssoc.staMuAssocLabel = 'tpToStaTrackAssociation'
RecoMuonVTrackAssoc.glbMuAssocLabel = 'tpToGlbTrackAssociation'

muonValidation_seq = cms.Sequence(RecoMuonVMuAssoc+RecoMuonVTrackAssoc)
