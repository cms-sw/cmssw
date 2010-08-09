import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidation_cff import *

# Configurations for MuonTrackValidators

trkMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.trkMuonTrackVTrackAssoc.clone()
trkMuonTrackVTrackAssocFS.associatormap = 'tpToTkmuTrackAssociationFS'

staMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.staMuonTrackVTrackAssoc.clone()
staMuonTrackVTrackAssocFS.associatormap = 'tpToStaTrackAssociationFS'
staMuonTrackVTrackAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staMuonTrackVTrackAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

staUpdMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.staUpdMuonTrackVTrackAssoc.clone()
staUpdMuonTrackVTrackAssocFS.associatormap = 'tpToStaUpdTrackAssociationFS'
staUpdMuonTrackVTrackAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staUpdMuonTrackVTrackAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

glbMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.glbMuonTrackVTrackAssoc.clone()
glbMuonTrackVTrackAssocFS.associatormap = 'tpToGlbTrackAssociationFS'
glbMuonTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
glbMuonTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

tevMuonFirstTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.tevMuonFirstTrackVTrackAssoc.clone()
tevMuonFirstTrackVTrackAssocFS.associatormap = 'tpToTevFirstTrackAssociationFS'
tevMuonFirstTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonFirstTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

tevMuonPickyTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.tevMuonPickyTrackVTrackAssoc.clone()
tevMuonPickyTrackVTrackAssocFS.associatormap = 'tpToTevPickyTrackAssociationFS'
tevMuonPickyTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonPickyTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

tevMuonDytTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.tevMuonDytTrackVTrackAssoc.clone()
tevMuonDytTrackVTrackAssocFS.associatormap = 'tpToTevDytTrackAssociationFS'
tevMuonDytTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonDytTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

staMuonTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.staMuonTrackVMuonAssoc.clone()
staMuonTrackVMuonAssocFS.associatormap = 'tpToStaMuonAssociationFS'
staMuonTrackVMuonAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staMuonTrackVMuonAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

staUpdMuonTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.staUpdMuonTrackVMuonAssoc.clone()
staUpdMuonTrackVMuonAssocFS.associatormap = 'tpToStaUpdMuonAssociationFS'
staUpdMuonTrackVMuonAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staUpdMuonTrackVMuonAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

glbMuonTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.glbMuonTrackVMuonAssoc.clone()
glbMuonTrackVMuonAssocFS.associatormap = 'tpToGlbMuonAssociationFS'
glbMuonTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
glbMuonTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

tevMuonFirstTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.tevMuonFirstTrackVMuonAssoc.clone()
tevMuonFirstTrackVMuonAssocFS.associatormap = 'tpToTevFirstMuonAssociationFS'
tevMuonFirstTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonFirstTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

tevMuonPickyTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.tevMuonPickyTrackVMuonAssoc.clone()
tevMuonPickyTrackVMuonAssocFS.associatormap = 'tpToTevPickyMuonAssociationFS'
tevMuonPickyTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonPickyTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

tevMuonDytTrackVMuonAssocFS = Validation.RecoMuon.muonValidation_cff.tevMuonDytTrackVMuonAssoc.clone()
tevMuonDytTrackVMuonAssocFS.associatormap = 'tpToTevDytMuonAssociationFS'
tevMuonDytTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonDytTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

# Configurations for RecoMuonValidators

recoMuonVMuAssocFS = Validation.RecoMuon.muonValidation_cff.recoMuonVMuAssoc.clone()
recoMuonVMuAssocFS.trkMuAssocLabel = 'tpToTkMuonAssociationFS'
recoMuonVMuAssocFS.staMuAssocLabel = 'tpToStaUpdMuonAssociationFS'
recoMuonVMuAssocFS.glbMuAssocLabel = 'tpToGlbMuonAssociationFS'

recoMuonVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.recoMuonVTrackAssoc.clone()
recoMuonVTrackAssocFS.trkMuAssocLabel = 'tpToTkmuTrackAssociationFS'
recoMuonVTrackAssocFS.staMuAssocLabel = 'tpToStaUpdTrackAssociationFS'
recoMuonVTrackAssocFS.glbMuAssocLabel = 'tpToGlbTrackAssociationFS'

# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkMuonTrackVTrackAssocFS
                                         +staMuonTrackVMuonAssocFS+staUpdMuonTrackVMuonAssocFS+glbMuonTrackVMuonAssocFS
                                         +tevMuonFirstTrackVMuonAssocFS+tevMuonPickyTrackVMuonAssocFS+tevMuonDytTrackVMuonAssocFS
                                         +recoMuonVMuAssocFS)


# The muon association and validation sequence
recoMuonAssociationFastSim = cms.Sequence(muonAssociationFastSim_seq)
recoMuonValidationFastSim = cms.Sequence(muonValidationFastSim_seq)

