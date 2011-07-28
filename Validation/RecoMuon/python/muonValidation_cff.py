import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkMuonTrackVTrackAssoc.label = ('generalTracks',)
trkMuonTrackVTrackAssoc.usetracker = True
trkMuonTrackVTrackAssoc.usemuon = False

trkCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkCosmicMuonTrackVTrackAssoc.associatormap = 'tpToTkCosmicTrackAssociation'
trkCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByHits',)
trkCosmicMuonTrackVTrackAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)
trkCosmicMuonTrackVTrackAssoc.usetracker = True
trkCosmicMuonTrackVTrackAssoc.usemuon = False

staMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVTrackAssoc.associatormap = 'tpToStaTrackAssociation'
staMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staMuonTrackVTrackAssoc.label = ('standAloneMuons',)
staMuonTrackVTrackAssoc.usetracker = False
staMuonTrackVTrackAssoc.usemuon = True

staUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaUpdTrackAssociation'
staUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staUpdMuonTrackVTrackAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVTrackAssoc.usetracker = False
staUpdMuonTrackVTrackAssoc.usemuon = True

glbMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVTrackAssoc.associatormap = 'tpToGlbTrackAssociation'
glbMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbMuonTrackVTrackAssoc.label = ('globalMuons',)
glbMuonTrackVTrackAssoc.usetracker = True
glbMuonTrackVTrackAssoc.usemuon = True

staSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVTrackAssoc.associatormap = 'tpToStaSETTrackAssociation'
staSETMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staSETMuonTrackVTrackAssoc.label = ('standAloneSETMuons',)
staSETMuonTrackVTrackAssoc.usetracker = False
staSETMuonTrackVTrackAssoc.usemuon = True

staSETUpdMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVTrackAssoc.associatormap = 'tpToStaSETUpdTrackAssociation'
staSETUpdMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staSETUpdMuonTrackVTrackAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)
staSETUpdMuonTrackVTrackAssoc.usetracker = False
staSETUpdMuonTrackVTrackAssoc.usemuon = True

glbSETMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVTrackAssoc.associatormap = 'tpToGlbSETTrackAssociation'
glbSETMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbSETMuonTrackVTrackAssoc.label = ('globalSETMuons',)
glbSETMuonTrackVTrackAssoc.usetracker = True
glbSETMuonTrackVTrackAssoc.usemuon = True

tevMuonFirstTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVTrackAssoc.associatormap = 'tpToTevFirstTrackAssociation'
tevMuonFirstTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonFirstTrackVTrackAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVTrackAssoc.usetracker = True
tevMuonFirstTrackVTrackAssoc.usemuon = True

tevMuonPickyTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVTrackAssoc.associatormap = 'tpToTevPickyTrackAssociation'
tevMuonPickyTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonPickyTrackVTrackAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVTrackAssoc.usetracker = True
tevMuonPickyTrackVTrackAssoc.usemuon = True

tevMuonDytTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVTrackAssoc.associatormap = 'tpToTevDytTrackAssociation'
tevMuonDytTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
tevMuonDytTrackVTrackAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVTrackAssoc.usetracker = True
tevMuonDytTrackVTrackAssoc.usemuon = True

staCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVTrackAssoc.associatormap = 'tpToStaCosmicTrackAssociation'
staCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
staCosmicMuonTrackVTrackAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVTrackAssoc.usetracker = False
staCosmicMuonTrackVTrackAssoc.usemuon = True

glbCosmicMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVTrackAssoc.associatormap = 'tpToGlbCosmicTrackAssociation'
glbCosmicMuonTrackVTrackAssoc.associators = ('TrackAssociatorByDeltaR',)
glbCosmicMuonTrackVTrackAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVTrackAssoc.usetracker = True
glbCosmicMuonTrackVTrackAssoc.usemuon = True

staMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)
staMuonTrackVMuonAssoc.usetracker = False
staMuonTrackVMuonAssoc.usemuon = True

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVMuonAssoc.usetracker = False
staUpdMuonTrackVMuonAssoc.usemuon = True

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbMuonTrackVMuonAssoc.label = ('extractedGlobalMuons',)
glbMuonTrackVMuonAssoc.usetracker = True
glbMuonTrackVMuonAssoc.usemuon = True

staSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETMuonTrackVMuonAssoc.associatormap = 'tpToStaSETMuonAssociation'
staSETMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staSETMuonTrackVMuonAssoc.label = ('standAloneSETMuons',)
staSETMuonTrackVMuonAssoc.usetracker = False
staSETMuonTrackVMuonAssoc.usemuon = True

staSETUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSETUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaSETUpdMuonAssociation'
staSETUpdMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staSETUpdMuonTrackVMuonAssoc.label = ('standAloneSETMuons:UpdatedAtVtx',)
staSETUpdMuonTrackVMuonAssoc.usetracker = False
staSETUpdMuonTrackVMuonAssoc.usemuon = True

glbSETMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbSETMuonTrackVMuonAssoc.associatormap = 'tpToGlbSETMuonAssociation'
glbSETMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbSETMuonTrackVMuonAssoc.label = ('globalSETMuons',)
glbSETMuonTrackVMuonAssoc.usetracker = True
glbSETMuonTrackVMuonAssoc.usemuon = True

tevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVMuonAssoc.associatormap = 'tpToTevFirstMuonAssociation'
tevMuonFirstTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVMuonAssoc.usetracker = True
tevMuonFirstTrackVMuonAssoc.usemuon = True

tevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVMuonAssoc.associatormap = 'tpToTevPickyMuonAssociation'
tevMuonPickyTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVMuonAssoc.usetracker = True
tevMuonPickyTrackVMuonAssoc.usemuon = True

tevMuonDytTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVMuonAssoc.associatormap = 'tpToTevDytMuonAssociation'
tevMuonDytTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
tevMuonDytTrackVMuonAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVMuonAssoc.usetracker = True
tevMuonDytTrackVMuonAssoc.usemuon = True

staCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVMuonAssoc.associatormap = 'tpToStaCosmicMuonAssociation'
staCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
staCosmicMuonTrackVMuonAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVMuonAssoc.usetracker = False
staCosmicMuonTrackVMuonAssoc.usemuon = True

glbCosmicMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVMuonAssoc.associatormap = 'tpToGlbCosmicMuonAssociation'
glbCosmicMuonTrackVMuonAssoc.associators = ('MuonAssociationByHits',)
glbCosmicMuonTrackVMuonAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVMuonAssoc.usetracker = True
glbCosmicMuonTrackVMuonAssoc.usemuon = True


# Configurations for RecoMuonValidators
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

import Validation.RecoMuon.RecoMuonValidator_cfi

recoMuonVMuAssoc = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc'
recoMuonVMuAssoc.trkMuLabel = 'generalTracks'
recoMuonVMuAssoc.staMuLabel = 'standAloneMuons:UpdatedAtVtx'
recoMuonVMuAssoc.glbMuLabel = 'extractedGlobalMuons'
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
muonValidation_seq = cms.Sequence(trkMuonTrackVTrackAssoc
                                 +staMuonTrackVMuonAssoc+staUpdMuonTrackVMuonAssoc+glbMuonTrackVMuonAssoc
                                 +recoMuonVMuAssoc)

muonValidationTEV_seq = cms.Sequence(tevMuonFirstTrackVMuonAssoc+tevMuonPickyTrackVMuonAssoc+tevMuonDytTrackVMuonAssoc)

muonValidationSET_seq = cms.Sequence(staSETMuonTrackVMuonAssoc+staSETUpdMuonTrackVMuonAssoc+glbSETMuonTrackVMuonAssoc)

muonValidationCosmic_seq = cms.Sequence(trkCosmicMuonTrackVTrackAssoc
                                 +staCosmicMuonTrackVMuonAssoc+glbCosmicMuonTrackVMuonAssoc)

# The muon association and validation sequence
recoMuonValidation = cms.Sequence((muonAssociation_seq*muonValidation_seq)
                                  +(muonAssociationTEV_seq*muonValidationTEV_seq)
                                  +(muonAssociationSET_seq*muonValidationSET_seq)
                                  )

recoCosmicMuonValidation = cms.Sequence(muonAssociationCosmic_seq*muonValidationCosmic_seq)
