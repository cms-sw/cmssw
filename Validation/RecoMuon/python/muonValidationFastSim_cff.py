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
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

from SimGeneral.MixingModule.mixNoPU_cfi                          import *
from SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi   import *
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

#tracker
muonAssociatorByHitsESProducerNoSimHits_trkFS = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_trkFS.ComponentName = 'muonAssociatorByHits_NoSimHits_tracker'
muonAssociatorByHitsESProducerNoSimHits_trkFS.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_trkFS.UseMuon  = False
recoMuonVMuAssoc_trkFS = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_trkFS.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Trk'
recoMuonVMuAssoc_trkFS.muonLabel = 'muons'
recoMuonVMuAssoc_trkFS.simLabel = 'mergedtruth:MergedTrackTruth'
recoMuonVMuAssoc_trkFS.muAssocLabel = 'muonAssociatorByHits_NoSimHits_tracker'
recoMuonVMuAssoc_trkFS.trackType = 'inner'
recoMuonVMuAssoc_trkFS.muonSelection = 'isTrackerMuon'

recoMuonVTrackAssoc_trkFS = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVTrackAssoc_trkFS.subDir = 'Muons/RecoMuonV/RecoMuon_TrackAssoc_Trk'
recoMuonVTrackAssoc_trkFS.muonLabel = 'muons'
recoMuonVTrackAssoc_trkFS.simLabel = 'mergedtruth:MergedTrackTruth'
recoMuonVTrackAssoc_trkFS.muAssocLabel = 'muonAssociatorByHits_NoSimHits_tracker'
recoMuonVMuAssoc_trkFS.trackType = 'inner'
recoMuonVMuAssoc_trkFS.muonSelection = 'isTrackerMuon'

#standalone
muonAssociatorByHitsESProducerNoSimHits_staFS = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_staFS.ComponentName = 'muonAssociatorByHits_NoSimHits_standalone'
muonAssociatorByHitsESProducerNoSimHits_staFS.UseTracker = False
muonAssociatorByHitsESProducerNoSimHits_staFS.UseMuon  = True
recoMuonVMuAssoc_staFS = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_staFS.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Sta'
recoMuonVMuAssoc_staFS.muonLabel = 'muons'
recoMuonVMuAssoc_staFS.simLabel = 'mergedtruth:MergedTrackTruth'
recoMuonVMuAssoc_staFS.muAssocLabel = 'muonAssociatorByHits_NoSimHits_standalone'
recoMuonVMuAssoc_staFS.trackType = 'outer'
recoMuonVMuAssoc_staFS.muonSelection = 'isStandAloneMuon'

recoMuonVTrackAssoc_staFS = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVTrackAssoc_staFS.subDir = 'Muons/RecoMuonV/RecoMuon_TrackAssoc_Sta'
recoMuonVTrackAssoc_staFS.muonLabel = 'muons'
recoMuonVTrackAssoc_staFS.simLabel = 'mergedtruth:MergedTrackTruth'
recoMuonVTrackAssoc_staFS.muAssocLabel = 'muonAssociatorByHits_NoSimHits_stadalone'
recoMuonVMuAssoc_staFS.trackType = 'outer'
recoMuonVMuAssoc_staFS.muonSelection = 'isStandAloneMuon'

#global
muonAssociatorByHitsESProducerNoSimHits_glbFS = SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi.muonAssociatorByHitsESProducerNoSimHits.clone()
muonAssociatorByHitsESProducerNoSimHits_glbFS.ComponentName = 'muonAssociatorByHits_NoSimHits_global'
muonAssociatorByHitsESProducerNoSimHits_glbFS.UseTracker = True
muonAssociatorByHitsESProducerNoSimHits_glbFS.UseMuon  = True
recoMuonVMuAssoc_glbFS = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_glbFS.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Glb'
recoMuonVMuAssoc_glbFS.muonLabel = 'muons'
recoMuonVMuAssoc_glbFS.simLabel = 'mergedtruth:MergedTrackTruth'
recoMuonVMuAssoc_glbFS.muAssocLabel = 'muonAssociatorByHits_NoSimHits_global'
recoMuonVMuAssoc_glbFS.trackType = 'global'
recoMuonVMuAssoc_glbFS.muonSelection = 'isGlobalMuon'

recoMuonVTrackAssoc_glbFS = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVTrackAssoc_glbFS.subDir = 'Muons/RecoMuonV/RecoMuon_TrackAssoc_Glb'
recoMuonVTrackAssoc_glbFS.muonLabel = 'muons'
recoMuonVTrackAssoc_glbFS.simLabel = 'mergedtruth:MergedTrackTruth'
recoMuonVTrackAssoc_glbFS.muAssocLabel = 'muonAssociatorByHits_NoSimHits_global'
recoMuonVMuAssoc_glbFS.trackType = 'global'
recoMuonVMuAssoc_glbFS.muonSelection = 'isGlobalMuon'

# Muon validation sequence
muonValidationFastSim_seq = cms.Sequence(trkMuonTrackVTrackAssocFS
                                         +staMuonTrackVMuonAssocFS+staUpdMuonTrackVMuonAssocFS+glbMuonTrackVMuonAssocFS
                                         +tevMuonFirstTrackVMuonAssocFS+tevMuonPickyTrackVMuonAssocFS+tevMuonDytTrackVMuonAssocFS
                                         +recoMuonVMuAssoc_trkFS+recoMuonVMuAssoc_staFS+recoMuonVMuAssoc_glbFS)


# The muon association and validation sequence
recoMuonAssociationFastSim = cms.Sequence(muonAssociationFastSim_seq)
recoMuonValidationFastSim = cms.Sequence(mix+trackingParticlesNoSimHits+muonValidationFastSim_seq)
