import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociationFS'
trkMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkMuonTrackVTrackAssoc.label = ('generalTracks',)

staMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociationFS'
staMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociationFS'
glbMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbMuonTrackVTrackAssoc.label = ('globalMuons',)

staMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociationFS'
staMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociationFS'
glbMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVMuonAssoc.label = ('globalMuons',)

l2MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2MuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociationFS'
l2MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2MuonTrackV.beamSpot = 'offlineBeamSpot'
l2MuonTrackV.nintHit = 35
l2MuonTrackV.maxHit = 35.0
l2MuonTrackV.maxpT = 1100.0

l3MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonTrackV.associatormap = 'tpToL3TrackAssociationFS'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l3MuonTrackV.beamSpot = 'offlineBeamSpot'
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

recoMuonVMuAssoc.trkMuAssocLabel = 'tpToTkMuonAssociationFS'
recoMuonVMuAssoc.staMuAssocLabel = 'tpToStaMuonAssociationFS'
recoMuonVMuAssoc.glbMuAssocLabel = 'tpToGlbMuonAssociationFS'

recoMuonVTrackAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()

recoMuonVTrackAssoc.subDir = 'RecoMuonV/RecoMuon_TrackAssoc'

recoMuonVTrackAssoc.trkMuLabel = 'generalTracks'
recoMuonVTrackAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVTrackAssoc.glbMuLabel = 'globalMuons'

recoMuonVTrackAssoc.trkMuAssocLabel = 'tpToTkmuTrackAssociationFS'
recoMuonVTrackAssoc.staMuAssocLabel = 'tpToStaTrackAssociationFS'
recoMuonVTrackAssoc.glbMuAssocLabel = 'tpToGlbTrackAssociationFS'

# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkMuonTrackVTrackAssoc+staMuonTrackVTrackAssoc+glbMuonTrackVTrackAssoc
                                         +staMuonTrackVMuonAssoc+glbMuonTrackVMuonAssoc
                                         +recoMuonVMuAssoc+recoMuonVTrackAssoc)


# The muon association and validation sequence
recoMuonValidationFastSim = cms.Sequence(muonAssociationFastSim_seq*muonValidationFastSim_seq)

