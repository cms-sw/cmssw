import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkMuonTrackVTrackAssoc.label = ('generalTracks',)

trkCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkCosmicMuonTrackVTrackAssoc.associatormap = 'tpToTkCosmicTrackAssociation'
trkCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkCosmicMuonTrackVTrackAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)



staMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staMuonTrackVTrackAssoc.label = ('standAloneMuons',)

staUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociation'
staUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staUpdMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociation'
glbMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbMuonTrackVTrackAssoc.label = ('globalMuons',)

staSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVTrackAssoc.associatormap = 'tpToStaSETTrackAssociation'
staSETMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staSETMuonTrackVTrackAssoc.label = ('standAloneSETMuons',)

staSETUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaSETUpdTrackAssociation'
staSETUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staSETUpdMuonTrackVTrackAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)

glbSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVTrackAssoc.associatormap = 'tpToGlbSETTrackAssociation'
glbSETMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbSETMuonTrackVTrackAssoc.label = ('globalSETMuons',)

tevMuonFirstTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVTrackAssoc.associatormap = 'tpToTevFirstTrackAssociation'
tevMuonFirstTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonFirstTrackVTrackAssoc.label = ('tevMuons:firstHit',)

tevMuonPickyTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVTrackAssoc.associatormap = 'tpToTevPickyTrackAssociation'
tevMuonPickyTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonPickyTrackVTrackAssoc.label = ('tevMuons:picky',)

staCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVTrackAssoc.associatormap = 'tpToStaCosmicTrackAssociation'
staCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staCosmicMuonTrackVTrackAssoc.label = ('globalCosmicMuons',)

glbCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVTrackAssoc.associatormap = 'tpToGlbCosmicTrackAssociation'
glbCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbCosmicMuonTrackVTrackAssoc.label = ('cosmicMuons',)



staMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVMuonAssoc.label = ('globalMuons',)

staSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVMuonAssoc.associatormap = 'tpToStaSETMuonAssociation'
staSETMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staSETMuonTrackVMuonAssoc.label = ('standAloneSETMuons',)

staSETUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaSETUpdMuonAssociation'
staSETUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staSETUpdMuonTrackVMuonAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)

glbSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVMuonAssoc.associatormap = 'tpToGlbSETMuonAssociation'
glbSETMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbSETMuonTrackVMuonAssoc.label = ('globalSETMuons',)

tevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVMuonAssoc.associatormap = 'tpToTevFirstMuonAssociation'
tevMuonFirstTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)

tevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVMuonAssoc.associatormap = 'tpToTevPickyMuonAssociation'
tevMuonPickyTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)

staCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVMuonAssoc.associatormap = 'tpToStaCosmicMuonAssociation'
staCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staCosmicMuonTrackVMuonAssoc.label = ('cosmicMuons',)

glbCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
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

muonValidationSET_seq = cms.Sequence(staSETMuonTrackVTrackAssoc+staSETUpdMuonTrackVTrackAssoc+glbSETMuonTrackVTrackAssoc
                                     +staSETMuonTrackVMuonAssoc+staSETUpdMuonTrackVMuonAssoc+glbSETMuonTrackVMuonAssoc)

muonValidationCosmic_seq = cms.Sequence(trkCosmicMuonTrackVTrackAssoc+staCosmicMuonTrackVTrackAssoc+glbCosmicMuonTrackVTrackAssoc
                                 +staCosmicMuonTrackVMuonAssoc+glbCosmicMuonTrackVMuonAssoc)

# The muon association and validation sequence
recoMuonValidation = cms.Sequence((muonAssociation_seq*muonValidation_seq)
                                  +(muonAssociationTEV_seq*muonValidationTEV_seq)
                                  +(muonAssociationSET_seq*muonValidationSET_seq)
                                  )

recoCosmicMuonValidation = cms.Sequence(muonAssociationCosmic_seq*muonValidationCosmic_seq)
