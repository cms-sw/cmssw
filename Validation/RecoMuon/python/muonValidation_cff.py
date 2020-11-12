#
# Production configuration for FullSim: muon track validation using MuonAssociatorByHits
#
import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.track_selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *
import Validation.RecoMuon.MuonTrackValidator_cfi

from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *

from Validation.RecoMuon.RecoMuonValidator_cff import *

# quickTrackAssociatorByHits on probeTracks used as monitor wrt MuonAssociatorByHits

trkMuonTrackVTrackAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
trkMuonTrackVTrackAssoc.associators = ('trackAssociatorByHits',)
#trkMuonTrackVTrackAssoc.label = ('generalTracks',)
trkMuonTrackVTrackAssoc.label = ('probeTracks',)
trkMuonTrackVTrackAssoc.muonHistoParameters = trkMuonHistoParameters

# MuonAssociatorByHits used for all track collections

trkProbeTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkProbeTrackVMuonAssoc.associatormap = 'tpToTkMuonAssociation'
#trkProbeTrackVMuonAssoc.label = ('generalTracks',)
trkProbeTrackVMuonAssoc.label = ('probeTracks',)
trkProbeTrackVMuonAssoc.muonHistoParameters = trkMuonHistoParameters

staSeedTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staSeedTrackVMuonAssoc.associatormap = 'tpToStaSeedAssociation'
staSeedTrackVMuonAssoc.label = ('seedsOfSTAmuons',)
staSeedTrackVMuonAssoc.muonHistoParameters = staSeedMuonHistoParameters

staMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staMuonTrackVMuonAssoc.associatormap = 'tpToStaMuonAssociation'
staMuonTrackVMuonAssoc.label = ('standAloneMuons',)
staMuonTrackVMuonAssoc.muonHistoParameters = staMuonHistoParameters

staUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaUpdMuonAssociation'
staUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVMuonAssoc.muonHistoParameters = staUpdMuonHistoParameters

glbMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbMuonTrackVMuonAssoc.associatormap = 'tpToGlbMuonAssociation'
glbMuonTrackVMuonAssoc.label = ('globalMuons',)
glbMuonTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

staRefitMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staRefitMuonTrackVMuonAssoc.associatormap = 'tpToStaRefitMuonAssociation'
staRefitMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons',)
staRefitMuonTrackVMuonAssoc.muonHistoParameters = staMuonHistoParameters

staRefitUpdMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staRefitUpdMuonTrackVMuonAssoc.associatormap = 'tpToStaRefitUpdMuonAssociation'
staRefitUpdMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons:UpdatedAtVtx',)
staRefitUpdMuonTrackVMuonAssoc.muonHistoParameters = staUpdMuonHistoParameters

displacedTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
displacedTrackVMuonAssoc.associatormap = 'tpToDisplacedTrkMuonAssociation'
displacedTrackVMuonAssoc.label = ('displacedTracks',)
displacedTrackVMuonAssoc.muonTPSelector = displacedMuonTPSet
displacedTrackVMuonAssoc.muonHistoParameters = displacedTrkMuonHistoParameters

displacedStaSeedTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
displacedStaSeedTrackVMuonAssoc.associatormap = 'tpToDisplacedStaSeedAssociation'
displacedStaSeedTrackVMuonAssoc.label = ('seedsOfDisplacedSTAmuons',)
displacedStaSeedTrackVMuonAssoc.muonTPSelector = displacedMuonTPSet
displacedStaSeedTrackVMuonAssoc.muonHistoParameters = displacedStaSeedMuonHistoParameters

displacedStaMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
displacedStaMuonTrackVMuonAssoc.associatormap = 'tpToDisplacedStaMuonAssociation'
displacedStaMuonTrackVMuonAssoc.label = ('displacedStandAloneMuons',)
displacedStaMuonTrackVMuonAssoc.muonTPSelector = displacedMuonTPSet
displacedStaMuonTrackVMuonAssoc.muonHistoParameters = displacedStaMuonHistoParameters

displacedGlbMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
displacedGlbMuonTrackVMuonAssoc.associatormap = 'tpToDisplacedGlbMuonAssociation'
displacedGlbMuonTrackVMuonAssoc.label = ('displacedGlobalMuons',)
displacedGlbMuonTrackVMuonAssoc.muonTPSelector = displacedMuonTPSet
displacedGlbMuonTrackVMuonAssoc.muonHistoParameters = displacedGlbMuonHistoParameters

tevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonFirstTrackVMuonAssoc.associatormap = 'tpToTevFirstMuonAssociation'
tevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)
tevMuonFirstTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

tevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonPickyTrackVMuonAssoc.associatormap = 'tpToTevPickyMuonAssociation'
tevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)
tevMuonPickyTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

tevMuonDytTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tevMuonDytTrackVMuonAssoc.associatormap = 'tpToTevDytMuonAssociation'
tevMuonDytTrackVMuonAssoc.label = ('tevMuons:dyt',)
tevMuonDytTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

tunepMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
tunepMuonTrackVMuonAssoc.associatormap = 'tpToTunePMuonAssociation'
tunepMuonTrackVMuonAssoc.label = ('tunepMuonTracks',)
tunepMuonTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

pfMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
pfMuonTrackVMuonAssoc.associatormap = 'tpToPFMuonAssociation'
pfMuonTrackVMuonAssoc.label = ('pfMuonTracks',)
pfMuonTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

recomuMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
recomuMuonTrackVMuonAssoc.associatormap = 'tpTorecoMuonMuonAssociation'
recomuMuonTrackVMuonAssoc.label = ('recoMuonTracks',)
recomuMuonTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

gemMuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
gemMuonTrackVMuonAssoc.associatormap = 'tpToGEMMuonMuonAssociation'
gemMuonTrackVMuonAssoc.label = ('extractGemMuons',)
gemMuonTrackVMuonAssoc.muonHistoParameters = gemMuonHistoParameters

me0MuonTrackVMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
me0MuonTrackVMuonAssoc.associatormap = 'tpToME0MuonMuonAssociation'
me0MuonTrackVMuonAssoc.label = ('extractMe0Muons',)
me0MuonTrackVMuonAssoc.muonTPSelector = me0MuonTPSet
me0MuonTrackVMuonAssoc.muonHistoParameters = me0MuonHistoParameters

# cosmics 2-leg reco
trkCosmicMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkCosmicMuonTrackVSelMuonAssoc.associatormap = 'tpToTkCosmicSelMuonAssociation'
trkCosmicMuonTrackVSelMuonAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)
trkCosmicMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
trkCosmicMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
trkCosmicMuonTrackVSelMuonAssoc.BiDirectional_RecoToSim_association = False
trkCosmicMuonTrackVSelMuonAssoc.muonHistoParameters = trkCosmicMuonHistoParameters

staCosmicMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmicMuonTrackVSelMuonAssoc.associatormap = 'tpToStaCosmicSelMuonAssociation'
staCosmicMuonTrackVSelMuonAssoc.label = ('cosmicMuons',)
staCosmicMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
staCosmicMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
staCosmicMuonTrackVSelMuonAssoc.BiDirectional_RecoToSim_association = False
staCosmicMuonTrackVSelMuonAssoc.muonHistoParameters = staCosmicMuonHistoParameters

glbCosmicMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmicMuonTrackVSelMuonAssoc.associatormap = 'tpToGlbCosmicSelMuonAssociation'
glbCosmicMuonTrackVSelMuonAssoc.label = ('globalCosmicMuons',)
glbCosmicMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
glbCosmicMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
glbCosmicMuonTrackVSelMuonAssoc.BiDirectional_RecoToSim_association = False
glbCosmicMuonTrackVSelMuonAssoc.muonHistoParameters = glbCosmicMuonHistoParameters

# cosmics 1-leg reco
trkCosmic1LegMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
trkCosmic1LegMuonTrackVSelMuonAssoc.associatormap = 'tpToTkCosmic1LegSelMuonAssociation'
trkCosmic1LegMuonTrackVSelMuonAssoc.label = ('ctfWithMaterialTracksP5',)
trkCosmic1LegMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
trkCosmic1LegMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
trkCosmic1LegMuonTrackVSelMuonAssoc.muonHistoParameters = trkCosmic1LegMuonHistoParameters

staCosmic1LegMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
staCosmic1LegMuonTrackVSelMuonAssoc.associatormap = 'tpToStaCosmic1LegSelMuonAssociation'
staCosmic1LegMuonTrackVSelMuonAssoc.label = ('cosmicMuons1Leg',)
staCosmic1LegMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
staCosmic1LegMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
staCosmic1LegMuonTrackVSelMuonAssoc.muonHistoParameters = staCosmic1LegMuonHistoParameters

glbCosmic1LegMuonTrackVSelMuonAssoc = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
glbCosmic1LegMuonTrackVSelMuonAssoc.associatormap = 'tpToGlbCosmic1LegSelMuonAssociation'
glbCosmic1LegMuonTrackVSelMuonAssoc.label = ('globalCosmicMuons1Leg',)
glbCosmic1LegMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
glbCosmic1LegMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
glbCosmic1LegMuonTrackVSelMuonAssoc.muonHistoParameters = glbCosmic1LegMuonHistoParameters

##################################################################################
# Muon validation sequences using MuonTrackValidator
#
muonValidation_seq = cms.Sequence(
    probeTracks_seq + tpToTkMuonAssociation + trkProbeTrackVMuonAssoc
    +trackAssociatorByHits + tpToTkmuTrackAssociation + trkMuonTrackVTrackAssoc
    +seedsOfSTAmuons_seq + tpToStaSeedAssociation + staSeedTrackVMuonAssoc
    +tpToStaMuonAssociation + staMuonTrackVMuonAssoc
    +tpToStaUpdMuonAssociation + staUpdMuonTrackVMuonAssoc
    +tpToGlbMuonAssociation + glbMuonTrackVMuonAssoc
    +pfMuonTracks_seq + tpToPFMuonAssociation + pfMuonTrackVMuonAssoc
    +recoMuonTracks_seq + tpTorecoMuonMuonAssociation + recomuMuonTrackVMuonAssoc
)

muonValidationTEV_seq = cms.Sequence(
    tpToTevFirstMuonAssociation + tevMuonFirstTrackVMuonAssoc
    +tpToTevPickyMuonAssociation + tevMuonPickyTrackVMuonAssoc
    +tpToTevDytMuonAssociation + tevMuonDytTrackVMuonAssoc
    +tunepMuonTracks_seq + tpToTunePMuonAssociation + tunepMuonTrackVMuonAssoc
)

muonValidationRefit_seq = cms.Sequence(
    tpToStaRefitMuonAssociation + staRefitMuonTrackVMuonAssoc
    +tpToStaRefitUpdMuonAssociation + staRefitUpdMuonTrackVMuonAssoc
)

muonValidationDisplaced_seq = cms.Sequence(
    seedsOfDisplacedSTAmuons_seq + tpToDisplacedStaSeedAssociation + displacedStaSeedTrackVMuonAssoc
    +tpToDisplacedStaMuonAssociation + displacedStaMuonTrackVMuonAssoc
    +tpToDisplacedTrkMuonAssociation + displacedTrackVMuonAssoc
    +tpToDisplacedGlbMuonAssociation + displacedGlbMuonTrackVMuonAssoc
)

muonValidationCosmic_seq = cms.Sequence(
    tpToTkCosmicSelMuonAssociation + trkCosmicMuonTrackVSelMuonAssoc
    +tpToTkCosmic1LegSelMuonAssociation + trkCosmic1LegMuonTrackVSelMuonAssoc
    +tpToStaCosmicSelMuonAssociation + staCosmicMuonTrackVSelMuonAssoc
    +tpToStaCosmic1LegSelMuonAssociation + staCosmic1LegMuonTrackVSelMuonAssoc
    +tpToGlbCosmicSelMuonAssociation + glbCosmicMuonTrackVSelMuonAssoc
    +tpToGlbCosmic1LegSelMuonAssociation + glbCosmic1LegMuonTrackVSelMuonAssoc
)

recoMuonValidation_reduced_seq = cms.Sequence(
    probeTracks_seq + tpToTkMuonAssociation + trkProbeTrackVMuonAssoc
    +tpToStaUpdMuonAssociation + staUpdMuonTrackVMuonAssoc
    +tpToGlbMuonAssociation + glbMuonTrackVMuonAssoc
    +tunepMuonTracks_seq + tpToTunePMuonAssociation + tunepMuonTrackVMuonAssoc
    +pfMuonTracks_seq + tpToPFMuonAssociation + pfMuonTrackVMuonAssoc
    +recoMuonTracks_seq + tpTorecoMuonMuonAssociation + recomuMuonTrackVMuonAssoc
    +tpToDisplacedStaMuonAssociation + displacedStaMuonTrackVMuonAssoc
    +tpToDisplacedTrkMuonAssociation + displacedTrackVMuonAssoc
    +tpToDisplacedGlbMuonAssociation + displacedGlbMuonTrackVMuonAssoc
)

gemMuonValidation = cms.Sequence(extractGemMuonsTracks_seq + tpToGEMMuonMuonAssociation + gemMuonTrackVMuonAssoc)
me0MuonValidation = cms.Sequence(extractMe0MuonsTracks_seq + tpToME0MuonMuonAssociation + me0MuonTrackVMuonAssoc)

##########################################################################
# The full offline muon validation sequence
#
recoMuonValidation = cms.Sequence(
    muonValidation_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationDisplaced_seq + muonValidationRMV_seq
    )

# no displaced muons in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(recoMuonValidation, cms.Sequence(muonValidation_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationRMV_seq))

# sequence for cosmic muons
recoCosmicMuonValidation = cms.Sequence(
    muonValidationCosmic_seq
    )

# sequences for muon upgrades
#
_run3_muonValidation = recoMuonValidation.copy()              #For full validation
#_run3_muonValidation = recoMuonValidation_reduced_seq.copy()
_run3_muonValidation += gemMuonValidation

#_phase2_muonValidation = recoMuonValidation.copy()              #For full validation 
_phase2_muonValidation = recoMuonValidation_reduced_seq.copy()
_phase2_muonValidation += gemMuonValidation
_phase2_muonValidation += me0MuonValidation

#_phase2_ge0_muonValidation = recoMuonValidation.copy()          #For full validation
_phase2_ge0_muonValidation = recoMuonValidation_reduced_seq.copy()
_phase2_ge0_muonValidation += gemMuonValidation

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( recoMuonValidation, _run3_muonValidation )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( recoMuonValidation, _phase2_muonValidation )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toReplaceWith( recoMuonValidation, _phase2_ge0_muonValidation )
