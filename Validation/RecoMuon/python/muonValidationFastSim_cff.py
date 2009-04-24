import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidation_cff import *

# Configurations for MultiTrackValidators

trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociationFS'

staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociationFS'

staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociationFS'

glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociationFS'

staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociationFS'

staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociationFS'

glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociationFS'

# Configurations for RecoMuonValidators

recoMuonVMuAssoc.trkMuLabel = 'generalTracks'
recoMuonVMuAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVMuAssoc.glbMuLabel = 'globalMuons'

recoMuonVMuAssoc.trkMuAssocLabel = 'tpToTkMuonAssociationFS'
recoMuonVMuAssoc.staMuAssocLabel = 'tpToStaUpdMuonAssociationFS'
recoMuonVMuAssoc.glbMuAssocLabel = 'tpToGlbMuonAssociationFS'

recoMuonVTrackAssoc.trkMuAssocLabel = 'tpToTkmuTrackAssociationFS'
recoMuonVTrackAssoc.staMuAssocLabel = 'tpToStaUpdTrackAssociationFS'
recoMuonVTrackAssoc.glbMuAssocLabel = 'tpToGlbTrackAssociationFS'

# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkMuonTrackVTrackAssoc+staMuonTrackVTrackAssoc+staUpdMuonTrackVTrackAssoc+glbMuonTrackVTrackAssoc
                                         +staMuonTrackVMuonAssoc+staUpdMuonTrackVMuonAssoc+glbMuonTrackVMuonAssoc
                                         +recoMuonVMuAssoc+recoMuonVTrackAssoc)


# The muon association and validation sequence
recoMuonValidationFastSim = cms.Sequence(muonAssociationFastSim_seq*muonValidationFastSim_seq)

