import FWCore.ParameterSet.Config as cms

# Configurations for MuonTrackValidators

from Validation.RecoMuon.muonValidation_cff import trkMuonTrackVTrackAssoc

#trkMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.trkMuonTrackVTrackAssoc.clone()
trkMuonTrackVTrackAssocFS = trkMuonTrackVTrackAssoc.clone()
trkMuonTrackVTrackAssocFS.associatormap = 'tpToTkmuTrackAssociationFS'

from Validation.RecoMuon.muonValidation_cff import staMuonTrackVTrackAssoc

#staMuonTrackVTrackAssocFS = Validation.RecoMuon.muonValidation_cff.staMuonTrackVTrackAssoc.clone()
staMuonTrackVTrackAssocFS = staMuonTrackVTrackAssoc.clone()
staMuonTrackVTrackAssocFS.associatormap = 'tpToStaTrackAssociationFS'
staMuonTrackVTrackAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staMuonTrackVTrackAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staUpdMuonTrackVTrackAssoc

staUpdMuonTrackVTrackAssocFS = staUpdMuonTrackVTrackAssoc.clone()
staUpdMuonTrackVTrackAssocFS.associatormap = 'tpToStaUpdTrackAssociationFS'
staUpdMuonTrackVTrackAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staUpdMuonTrackVTrackAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import glbMuonTrackVTrackAssoc

glbMuonTrackVTrackAssocFS = glbMuonTrackVTrackAssoc.clone()
glbMuonTrackVTrackAssocFS.associatormap = 'tpToGlbTrackAssociationFS'
glbMuonTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
glbMuonTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonFirstTrackVTrackAssoc

tevMuonFirstTrackVTrackAssocFS = tevMuonFirstTrackVTrackAssoc.clone()
tevMuonFirstTrackVTrackAssocFS.associatormap = 'tpToTevFirstTrackAssociationFS'
tevMuonFirstTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonFirstTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonPickyTrackVTrackAssoc

tevMuonPickyTrackVTrackAssocFS = tevMuonPickyTrackVTrackAssoc.clone()
tevMuonPickyTrackVTrackAssocFS.associatormap = 'tpToTevPickyTrackAssociationFS'
tevMuonPickyTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonPickyTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonDytTrackVTrackAssoc

tevMuonDytTrackVTrackAssocFS = tevMuonDytTrackVTrackAssoc.clone()
tevMuonDytTrackVTrackAssocFS.associatormap = 'tpToTevDytTrackAssociationFS'
tevMuonDytTrackVTrackAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonDytTrackVTrackAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staMuonTrackVMuonAssoc

staMuonTrackVMuonAssocFS = staMuonTrackVMuonAssoc.clone()
staMuonTrackVMuonAssocFS.associatormap = 'tpToStaMuonAssociationFS'
staMuonTrackVMuonAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staMuonTrackVMuonAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staUpdMuonTrackVMuonAssoc

staUpdMuonTrackVMuonAssocFS = staUpdMuonTrackVMuonAssoc.clone()
staUpdMuonTrackVMuonAssocFS.associatormap = 'tpToStaUpdMuonAssociationFS'
staUpdMuonTrackVMuonAssocFS.label_tp_effic = 'mergedtruthMuon:MergedTrackTruth'
staUpdMuonTrackVMuonAssocFS.label_tp_fake = 'mergedtruthMuon:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import glbMuonTrackVMuonAssoc

glbMuonTrackVMuonAssocFS = glbMuonTrackVMuonAssoc.clone()
glbMuonTrackVMuonAssocFS.associatormap = 'tpToGlbMuonAssociationFS'
glbMuonTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
glbMuonTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonFirstTrackVMuonAssoc

tevMuonFirstTrackVMuonAssocFS = tevMuonFirstTrackVMuonAssoc.clone()
tevMuonFirstTrackVMuonAssocFS.associatormap = 'tpToTevFirstMuonAssociationFS'
tevMuonFirstTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonFirstTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonPickyTrackVMuonAssoc

tevMuonPickyTrackVMuonAssocFS = tevMuonPickyTrackVMuonAssoc.clone()
tevMuonPickyTrackVMuonAssocFS.associatormap = 'tpToTevPickyMuonAssociationFS'
tevMuonPickyTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonPickyTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonDytTrackVMuonAssoc

tevMuonDytTrackVMuonAssocFS = tevMuonDytTrackVMuonAssoc.clone()
tevMuonDytTrackVMuonAssocFS.associatormap = 'tpToTevDytMuonAssociationFS'
tevMuonDytTrackVMuonAssocFS.label_tp_effic = 'mergedtruth:MergedTrackTruth'
tevMuonDytTrackVMuonAssocFS.label_tp_fake = 'mergedtruth:MergedTrackTruth'

# Configurations for RecoMuonValidators
from Validation.RecoMuon.muonValidation_cff import *


# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkMuonTrackVTrackAssocFS
                                         +staMuonTrackVMuonAssocFS+staUpdMuonTrackVMuonAssocFS+glbMuonTrackVMuonAssocFS
                                         +tevMuonFirstTrackVMuonAssocFS+tevMuonPickyTrackVMuonAssocFS+tevMuonDytTrackVMuonAssocFS
                                         +recoMuonVMuAssoc_trk+recoMuonVMuAssoc_sta+recoMuonVMuAssoc_glb+recoMuonVMuAssoc_tgt
                                         +recoMuonVMuAssoc_trkPF+recoMuonVMuAssoc_staPF+recoMuonVMuAssoc_glbPF)

# The muon association and validation sequence
from Validation.RecoMuon.associators_cff import muonAssociationFastSim_seq
recoMuonAssociationFastSim = cms.Sequence(muonAssociationFastSim_seq)
recoMuonValidationFastSim = cms.Sequence(muonValidationFastSim_seq)

