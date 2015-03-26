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
staMuonTrackVTrackAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staMuonTrackVTrackAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staUpdMuonTrackVTrackAssoc

staUpdMuonTrackVTrackAssocFS = staUpdMuonTrackVTrackAssoc.clone()
staUpdMuonTrackVTrackAssocFS.associatormap = 'tpToStaUpdTrackAssociationFS'
staUpdMuonTrackVTrackAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staUpdMuonTrackVTrackAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import glbMuonTrackVTrackAssoc

glbMuonTrackVTrackAssocFS = glbMuonTrackVTrackAssoc.clone()
glbMuonTrackVTrackAssocFS.associatormap = 'tpToGlbTrackAssociationFS'
glbMuonTrackVTrackAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
glbMuonTrackVTrackAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonFirstTrackVTrackAssoc

tevMuonFirstTrackVTrackAssocFS = tevMuonFirstTrackVTrackAssoc.clone()
tevMuonFirstTrackVTrackAssocFS.associatormap = 'tpToTevFirstTrackAssociationFS'
tevMuonFirstTrackVTrackAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
tevMuonFirstTrackVTrackAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonPickyTrackVTrackAssoc

tevMuonPickyTrackVTrackAssocFS = tevMuonPickyTrackVTrackAssoc.clone()
tevMuonPickyTrackVTrackAssocFS.associatormap = 'tpToTevPickyTrackAssociationFS'
tevMuonPickyTrackVTrackAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
tevMuonPickyTrackVTrackAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonDytTrackVTrackAssoc

tevMuonDytTrackVTrackAssocFS = tevMuonDytTrackVTrackAssoc.clone()
tevMuonDytTrackVTrackAssocFS.associatormap = 'tpToTevDytTrackAssociationFS'
tevMuonDytTrackVTrackAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
tevMuonDytTrackVTrackAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import trkProbeTrackVMuonAssoc 
trkProbeTrackVMuonAssocFS = trkProbeTrackVMuonAssoc.clone()
trkProbeTrackVMuonAssocFS.associatormap = 'tpToTkMuonAssociationFS'
trkProbeTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
trkProbeTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staSeedTrackVMuonAssoc
staSeedTrackVMuonAssocFS = staSeedTrackVMuonAssoc.clone() 
staSeedTrackVMuonAssocFS.associatormap = 'tpToStaSeedAssociationFS' 
staSeedTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staSeedTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staMuonTrackVMuonAssoc

staMuonTrackVMuonAssocFS = staMuonTrackVMuonAssoc.clone()
staMuonTrackVMuonAssocFS.associatormap = 'tpToStaMuonAssociationFS'
staMuonTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staMuonTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staUpdMuonTrackVMuonAssoc

staUpdMuonTrackVMuonAssocFS = staUpdMuonTrackVMuonAssoc.clone()
staUpdMuonTrackVMuonAssocFS.associatormap = 'tpToStaUpdMuonAssociationFS'
staUpdMuonTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staUpdMuonTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staRefitMuonTrackVMuonAssoc

staRefitMuonTrackVMuonAssocFS = staRefitMuonTrackVMuonAssoc.clone()
staRefitMuonTrackVMuonAssocFS.associatormap = 'tpToStaRefitMuonAssociationFS'
staRefitMuonTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staRefitMuonTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import staRefitUpdMuonTrackVMuonAssoc

staRefitUpdMuonTrackVMuonAssocFS = staRefitUpdMuonTrackVMuonAssoc.clone()
staRefitUpdMuonTrackVMuonAssocFS.associatormap = 'tpToStaRefitUpdMuonAssociationFS'
staRefitUpdMuonTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
staRefitUpdMuonTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import glbMuonTrackVMuonAssoc

glbMuonTrackVMuonAssocFS = glbMuonTrackVMuonAssoc.clone()
glbMuonTrackVMuonAssocFS.associatormap = 'tpToGlbMuonAssociationFS'
glbMuonTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
glbMuonTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonFirstTrackVMuonAssoc

tevMuonFirstTrackVMuonAssocFS = tevMuonFirstTrackVMuonAssoc.clone()
tevMuonFirstTrackVMuonAssocFS.associatormap = 'tpToTevFirstMuonAssociationFS'
tevMuonFirstTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
tevMuonFirstTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonPickyTrackVMuonAssoc

tevMuonPickyTrackVMuonAssocFS = tevMuonPickyTrackVMuonAssoc.clone()
tevMuonPickyTrackVMuonAssocFS.associatormap = 'tpToTevPickyMuonAssociationFS'
tevMuonPickyTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
tevMuonPickyTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

from Validation.RecoMuon.muonValidation_cff import tevMuonDytTrackVMuonAssoc

tevMuonDytTrackVMuonAssocFS = tevMuonDytTrackVMuonAssoc.clone()
tevMuonDytTrackVMuonAssocFS.associatormap = 'tpToTevDytMuonAssociationFS'
tevMuonDytTrackVMuonAssocFS.label_tp_effic = 'mix:MergedTrackTruth'
tevMuonDytTrackVMuonAssocFS.label_tp_fake = 'mix:MergedTrackTruth'

# Configurations for RecoMuonValidators
from Validation.RecoMuon.muonValidation_cff import *


# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkProbeTrackVMuonAssocFS+trkMuonTrackVTrackAssocFS
                                         +staSeedTrackVMuonAssocFS
                                         +staMuonTrackVMuonAssocFS+staUpdMuonTrackVMuonAssocFS+glbMuonTrackVMuonAssocFS
                                         +staRefitMuonTrackVMuonAssocFS+staRefitUpdMuonTrackVMuonAssocFS
                                         +tevMuonFirstTrackVMuonAssocFS+tevMuonPickyTrackVMuonAssocFS+tevMuonDytTrackVMuonAssocFS
                                         +muonAssociatorByHitsNoSimHitsHelperTrk+muonAssociatorByHitsNoSimHitsHelperStandalone+muonAssociatorByHitsNoSimHitsHelperGlobal+muonAssociatorByHitsNoSimHitsHelperTight
                                         +recoMuonVMuAssoc_trk+recoMuonVMuAssoc_sta+recoMuonVMuAssoc_glb+recoMuonVMuAssoc_tgt
                                 +muonAssociatorByHitsNoSimHitsHelperTrkPF+muonAssociatorByHitsNoSimHitsHelperStandalonePF+muonAssociatorByHitsNoSimHitsHelperGlobalPF
                                         +recoMuonVMuAssoc_trkPF+recoMuonVMuAssoc_staPF+recoMuonVMuAssoc_glbPF)

# The muon association and validation sequence
from Validation.RecoMuon.associators_cff import muonAssociationFastSim_seq
recoMuonAssociationFastSim = cms.Sequence(muonAssociationFastSim_seq)
recoMuonValidationFastSim = cms.Sequence(muonValidationFastSim_seq)

