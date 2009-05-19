import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidation_cff import *

# Configurations for MultiTrackValidators

trkMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.trkMuonTrackVTrackAssoc.clone()
trkMuonTrackVTrackAssocFS.associatormap = 'tpToTkmuTrackAssociationFS'

staMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.staMuonTrackVTrackAssoc.clone()
staMuonTrackVTrackAssocFS.associatormap = 'tpToStaTrackAssociationFS'

staUpdMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.staUpdMuonTrackVTrackAssoc.clone()
staUpdMuonTrackVTrackAssocFS.associatormap = 'tpToStaUpdTrackAssociationFS'

glbMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.glbMuonTrackVTrackAssoc.clone()
glbMuonTrackVTrackAssocFS.associatormap = 'tpToGlbTrackAssociationFS'

staMuonTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.staMuonTrackVMuonAssoc.clone()
staMuonTrackVMuonAssocFS.associatormap = 'tpToStaMuonAssociationFS'

staUpdMuonTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.staUpdMuonTrackVMuonAssoc.clone()
staUpdMuonTrackVMuonAssocFS.associatormap = 'tpToStaUpdMuonAssociationFS'

glbMuonTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.glbMuonTrackVMuonAssoc.clone()
glbMuonTrackVMuonAssocFS.associatormap = 'tpToGlbMuonAssociationFS'

# Configurations for RecoMuonValidators

recoMuonVMuAssocFS = Validation.RecoMuon.muonValidation_cff.recoMuonVMuAssoc.clone()
recoMuonVMuAssocFS.trkMuLabel = 'generalTracks'
recoMuonVMuAssocFS.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVMuAssocFS.glbMuLabel = 'globalMuons'

recoMuonVMuAssocFS.trkMuAssocLabel = 'tpToTkMuonAssociationFS'
recoMuonVMuAssocFS.staMuAssocLabel = 'tpToStaUpdMuonAssociationFS'
recoMuonVMuAssocFS.glbMuAssocLabel = 'tpToGlbMuonAssociationFS'

recoMuonVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.recoMuonVTrackAssoc.clone()
recoMuonVTrackAssocFS.trkMuAssocLabel = 'tpToTkmuTrackAssociationFS'
recoMuonVTrackAssocFS.staMuAssocLabel = 'tpToStaUpdTrackAssociationFS'
recoMuonVTrackAssocFS.glbMuAssocLabel = 'tpToGlbTrackAssociationFS'

# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkMuonTrackVTrackAssocFS
                                         +staMuonTrackVTrackAssocFS+staUpdMuonTrackVTrackAssocFS+glbMuonTrackVTrackAssocFS
                                         +staMuonTrackVMuonAssocFS+staUpdMuonTrackVMuonAssocFS+glbMuonTrackVMuonAssocFS
                                         +recoMuonVMuAssocFS+recoMuonVTrackAssocFS)


# The muon association and validation sequence
recoMuonValidationFastSim = cms.Sequence(muonAssociationFastSim_seq*muonValidationFastSim_seq)

