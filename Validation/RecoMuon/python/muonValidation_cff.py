import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = 'TrackAssociatorByHits'
trkMuonTrackVTrackAssoc.label = ('generalTracks',)

staMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = 'TrackAssociatorByDeltaR'
staMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociation'
glbMuonTrackVTrackAssoc.associators = 'TrackAssociatorByDeltaR'
glbMuonTrackVTrackAssoc.label = ('globalMuons',)

staMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = 'MuonAssociationByHits'
staMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = 'MuonAssociationByHits'
glbMuonTrackVMuonAssoc.label = ('globalMuons',)

l2MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2MuonTrackV.label = 'hltL2Muons:UpdatedAtVtx'
l2MuonTrackV.associatormap = 'tpToL2TrackAssociation'
l2MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l2MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonTrackV.nintHit = 35
l2MuonTrackV.maxHit = 35.0
l2MuonTrackV.maxpT = 1100.0

l3MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonTrackV.associatormap = 'tpToL3TrackAssociation'
l3MuonTrackV.label = 'hltL3Muons'
l3MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l3MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonTrackV.nintHit = 35
l3MuonTrackV.maxHit = 35.0
l3MuonTrackV.maxpT = 1100.0

# Configurations for RecoMuonValidators
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

import Validation.RecoMuon.RecoMuonValidator_cfi

recoMuonVMuAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()

recoMuonVMuAssoc.subDir = 'RecoMuonV/RecoMuon_MuonAssoc'

recoMuonVMuAssoc.trkMuLabel = 'generalTracks'
recoMuonVMuAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVMuAssoc.glbMuLabel = 'globalMuons'

recoMuonVMuAssoc.trkMuAssocLabel = 'tpToTkMuonAssociation'
recoMuonVMuAssoc.staMuAssocLabel = 'tpToStaMuonAssociation'
recoMuonVMuAssoc.glbMuAssocLabel = 'tpToGlbMuonAssociation'

recoMuonVTrackAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()

recoMuonVTrackAssoc.subDir = 'RecoMuonV/RecoMuon_TrackAssoc'

recoMuonVTrackAssoc.trkMuLabel = 'generalTracks'
recoMuonVTrackAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVTrackAssoc.glbMuLabel = 'globalMuons'

recoMuonVTrackAssoc.trkMuAssocLabel = 'tpToTkmuTrackAssociation'
recoMuonVTrackAssoc.staMuAssocLabel = 'tpToStaTrackAssociation'
recoMuonVTrackAssoc.glbMuAssocLabel = 'tpToGlbTrackAssociation'

# Muon validation sequence
muonValidation_seq = cms.Sequence(trkMuonTrackVTrackAssoc+staMuonTrackVTrackAssoc+glbMuonTrackVTrackAssoc
                                 +staMuonTrackVMuonAssoc+glbMuonTrackVMuonAssoc
                                 +recoMuonVMuAssoc+recoMuonVTrackAssoc)

recoMuonValidation = cms.Sequence(muonAssociation_seq*muonValidation_seq)
