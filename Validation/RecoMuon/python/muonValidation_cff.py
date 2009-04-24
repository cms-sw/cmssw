import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkMuonTrackVTrackAssoc.label = ('generalTracks',)

staMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staMuonTrackVTrackAssoc.label = ('standAloneMuons',)

staUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociation'
staUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staUpdMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociation'
glbMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbMuonTrackVTrackAssoc.label = ('globalMuons',)

staMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVMuonAssoc.label = ('globalMuons',)


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
recoMuonVMuAssoc.staMuAssocLabel = 'tpToStaUpdMuonAssociation'
recoMuonVMuAssoc.glbMuAssocLabel = 'tpToGlbMuonAssociation'

recoMuonVTrackAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()

recoMuonVTrackAssoc.subDir = 'RecoMuonV/RecoMuon_TrackAssoc'

recoMuonVTrackAssoc.trkMuLabel = 'generalTracks'
recoMuonVTrackAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVTrackAssoc.glbMuLabel = 'globalMuons'

recoMuonVTrackAssoc.trkMuAssocLabel = 'tpToTkmuTrackAssociation'
recoMuonVTrackAssoc.staMuAssocLabel = 'tpToStaUpdTrackAssociation'
recoMuonVTrackAssoc.glbMuAssocLabel = 'tpToGlbTrackAssociation'

# Muon validation sequence
muonValidation_seq = cms.Sequence(trkMuonTrackVTrackAssoc+staMuonTrackVTrackAssoc+staUpdMuonTrackVTrackAssoc+glbMuonTrackVTrackAssoc
                                 +staMuonTrackVMuonAssoc+staUpdMuonTrackVMuonAssoc+glbMuonTrackVMuonAssoc
                                 +recoMuonVMuAssoc+recoMuonVTrackAssoc)


# The muon association and validation sequence
recoMuonValidation = cms.Sequence(muonAssociation_seq*muonValidation_seq)
