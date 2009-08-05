import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkMuonTrackVTrackAssoc.label = ('generalTracks',)

trkCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
trkCosmicMuonTrackVTrackAssoc.associatormap = 'tpToTkCosmicTrackAssociation'
trkCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkCosmicMuonTrackVTrackAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)



staMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staMuonTrackVTrackAssoc.label = ('standAloneMuons',)

staUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociation'
staUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staUpdMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociation'
glbMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbMuonTrackVTrackAssoc.label = ('globalMuons',)

tevMuonFirstTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
tevMuonFirstTrackVTrackAssoc.associatormap = 'tpToTevFirstTrackAssociation'
tevMuonFirstTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonFirstTrackVTrackAssoc.label = ('tevMuons:firstHit',)

tevMuonPickyTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
tevMuonPickyTrackVTrackAssoc.associatormap = 'tpToTevPickyTrackAssociation'
tevMuonPickyTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonPickyTrackVTrackAssoc.label = ('tevMuons:picky',)

staCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
staCosmicMuonTrackVTrackAssoc.associatormap = 'tpToStaCosmicTrackAssociation'
staCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staCosmicMuonTrackVTrackAssoc.label = ('globalCosmicMuons',)

glbCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
glbCosmicMuonTrackVTrackAssoc.associatormap = 'tpToGlbCosmicTrackAssociation'
glbCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbCosmicMuonTrackVTrackAssoc.label = ('cosmicMuons',)



staMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVMuonAssoc.label = ('globalMuons',)

tevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
tevMuonFirstTrackVMuonAssoc.associatormap = 'tpToTevFirstMuonAssociation'
tevMuonFirstTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)

tevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
tevMuonPickyTrackVMuonAssoc.associatormap = 'tpToTevPickyMuonAssociation'
tevMuonPickyTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)

staCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
staCosmicMuonTrackVMuonAssoc.associatormap = 'tpToStaCosmicMuonAssociation'
staCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staCosmicMuonTrackVMuonAssoc.label = ('cosmicMuons',)

glbCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()
glbCosmicMuonTrackVMuonAssoc.associatormap = 'tpToGlbCosmicMuonAssociation'
glbCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbCosmicMuonTrackVMuonAssoc.label = ('globalCosmicMuons',)



# Configurations for RecoMuonValidators
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

import Validation.RecoMuon.RecoMuonValidator_cfi

recoMuonVMuAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc'
recoMuonVMuAssoc.trkMuLabel = 'generalTracks'
recoMuonVMuAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVMuAssoc.glbMuLabel = 'globalMuons'
recoMuonVMuAssoc.trkMuAssocLabel = 'tpToTkMuonAssociation'
recoMuonVMuAssoc.staMuAssocLabel = 'tpToStaUpdMuonAssociation'
recoMuonVMuAssoc.glbMuAssocLabel = 'tpToGlbMuonAssociation'

recoMuonVTrackAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVTrackAssoc.subDir = 'Muons/RecoMuonV/RecoMuon_TrackAssoc'
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

muonValidationTEV_seq = cms.Sequence(tevMuonFirstTrackVTrackAssoc+tevMuonPickyTrackVTrackAssoc
                                    +tevMuonFirstTrackVMuonAssoc+tevMuonPickyTrackVMuonAssoc)

muonValidationCosmic_seq = cms.Sequence(trkCosmicMuonTrackVTrackAssoc+staCosmicMuonTrackVTrackAssoc+glbCosmicMuonTrackVTrackAssoc
                                 +staCosmicMuonTrackVMuonAssoc+glbCosmicMuonTrackVMuonAssoc)

# The muon association and validation sequence
recoMuonValidation = cms.Sequence(muonAssociation_seq*muonValidation_seq)
#recoMuonValidation = cms.Sequence((muonAssociation_seq*muonValidation_seq)+(muonAssociationTEV_seq*muonValidationTEV_seq))

recoCosmicMuonValidation = cms.Sequence(muonAssociationCosmic_seq*muonValidationCosmic_seq)
