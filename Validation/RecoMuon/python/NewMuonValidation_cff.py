#
# Production configuration for FullSim: muon track validation using MuonAssociatorByHits
#
import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.NewSelectors_cff import *
from Validation.RecoMuon.NewAssociators_cff import *
from Validation.RecoMuon.histoParameters_cff import *
import Validation.RecoMuon.NewMuonTrackValidator_cfi

from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *

from Validation.RecoMuon.NewRecoMuonValidator_cff import *

# quickTrackAssociatorByHits on probeTracks used as monitor wrt MuonAssociatorByHits

NEWtrkMuonTrackVTrackAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtrkMuonTrackVTrackAssoc.associatormap = 'NEWtpToTkmuTrackAssociation'
NEWtrkMuonTrackVTrackAssoc.associators = ('NEWtrackAssociatorByHits',)
#NEWtrkMuonTrackVTrackAssoc.label = ('generalTracks',)
NEWtrkMuonTrackVTrackAssoc.label = ('NEWprobeTracks',)
NEWtrkMuonTrackVTrackAssoc.muonHistoParameters = trkMuonHistoParameters

# MuonAssociatorByHits used for all track collections

NEWtrkProbeTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtrkProbeTrackVMuonAssoc.associatormap = 'NEWtpToTkMuonAssociation'
#trkProbeTrackVMuonAssoc.label = ('generalTracks',)
NEWtrkProbeTrackVMuonAssoc.label = ('NEWprobeTracks',)
NEWtrkProbeTrackVMuonAssoc.muonHistoParameters = trkMuonHistoParameters

NEWstaSeedTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaSeedTrackVMuonAssoc.associatormap = 'NEWtpToStaSeedAssociation'
NEWstaSeedTrackVMuonAssoc.label = ('NEWseedsOfSTAmuons',)
NEWstaSeedTrackVMuonAssoc.muonHistoParameters = staSeedMuonHistoParameters

NEWstaMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaMuonTrackVMuonAssoc.associatormap = 'NEWtpToStaMuonAssociation'
NEWstaMuonTrackVMuonAssoc.label = ('standAloneMuons',)
NEWstaMuonTrackVMuonAssoc.muonHistoParameters = staMuonHistoParameters

NEWstaUpdMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaUpdMuonTrackVMuonAssoc.associatormap = 'NEWtpToStaUpdMuonAssociation'
NEWstaUpdMuonTrackVMuonAssoc.label = ('standAloneMuons:UpdatedAtVtx',)
NEWstaUpdMuonTrackVMuonAssoc.muonHistoParameters = staUpdMuonHistoParameters

NEWglbMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWglbMuonTrackVMuonAssoc.associatormap = 'NEWtpToGlbMuonAssociation'
NEWglbMuonTrackVMuonAssoc.label = ('globalMuons',)
NEWglbMuonTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

NEWstaRefitMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaRefitMuonTrackVMuonAssoc.associatormap = 'NEWtpToStaRefitMuonAssociation'
NEWstaRefitMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons',)
NEWstaRefitMuonTrackVMuonAssoc.muonHistoParameters = staMuonHistoParameters

NEWstaRefitUpdMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaRefitUpdMuonTrackVMuonAssoc.associatormap = 'NEWtpToStaRefitUpdMuonAssociation'
NEWstaRefitUpdMuonTrackVMuonAssoc.label = ('refittedStandAloneMuons:UpdatedAtVtx',)
NEWstaRefitUpdMuonTrackVMuonAssoc.muonHistoParameters = staUpdMuonHistoParameters

NEWdisplacedTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWdisplacedTrackVMuonAssoc.associatormap = 'NEWtpToDisplacedTrkMuonAssociation'
NEWdisplacedTrackVMuonAssoc.label = ('displacedTracks',)
NEWdisplacedTrackVMuonAssoc.muonTPSelector = NewDisplacedMuonTPSet
NEWdisplacedTrackVMuonAssoc.muonHistoParameters = displacedTrkMuonHistoParameters

NEWdisplacedStaSeedTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWdisplacedStaSeedTrackVMuonAssoc.associatormap = 'NEWtpToDisplacedStaSeedAssociation'
NEWdisplacedStaSeedTrackVMuonAssoc.label = ('NEWseedsOfDisplacedSTAmuons',)
NEWdisplacedStaSeedTrackVMuonAssoc.muonTPSelector = NewDisplacedMuonTPSet
NEWdisplacedStaSeedTrackVMuonAssoc.muonHistoParameters = displacedStaSeedMuonHistoParameters

NEWdisplacedStaMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWdisplacedStaMuonTrackVMuonAssoc.associatormap = 'NEWtpToDisplacedStaMuonAssociation'
NEWdisplacedStaMuonTrackVMuonAssoc.label = ('displacedStandAloneMuons',)
NEWdisplacedStaMuonTrackVMuonAssoc.muonTPSelector = NewDisplacedMuonTPSet
NEWdisplacedStaMuonTrackVMuonAssoc.muonHistoParameters = displacedStaMuonHistoParameters

NEWdisplacedGlbMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWdisplacedGlbMuonTrackVMuonAssoc.associatormap = 'NEWtpToDisplacedGlbMuonAssociation'
NEWdisplacedGlbMuonTrackVMuonAssoc.label = ('displacedGlobalMuons',)
NEWdisplacedGlbMuonTrackVMuonAssoc.muonTPSelector = NewDisplacedMuonTPSet
NEWdisplacedGlbMuonTrackVMuonAssoc.muonHistoParameters = displacedGlbMuonHistoParameters

NEWtevMuonFirstTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtevMuonFirstTrackVMuonAssoc.associatormap = 'NEWtpToTevFirstMuonAssociation'
NEWtevMuonFirstTrackVMuonAssoc.label = ('tevMuons:firstHit',)
NEWtevMuonFirstTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

NEWtevMuonPickyTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtevMuonPickyTrackVMuonAssoc.associatormap = 'NEWtpToTevPickyMuonAssociation'
NEWtevMuonPickyTrackVMuonAssoc.label = ('tevMuons:picky',)
NEWtevMuonPickyTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

NEWtevMuonDytTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtevMuonDytTrackVMuonAssoc.associatormap = 'NEWtpToTevDytMuonAssociation'
NEWtevMuonDytTrackVMuonAssoc.label = ('tevMuons:dyt',)
NEWtevMuonDytTrackVMuonAssoc.muonHistoParameters = glbMuonHistoParameters

NEWgemMuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWgemMuonTrackVMuonAssoc.associatormap = 'NEWtpToGEMMuonMuonAssociation'
NEWgemMuonTrackVMuonAssoc.label = ('NEWextractGemMuons',)
NEWgemMuonTrackVMuonAssoc.muonHistoParameters = gemMuonHistoParameters

NEWme0MuonTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWme0MuonTrackVMuonAssoc.associatormap = 'NEWtpToME0MuonMuonAssociation'
NEWme0MuonTrackVMuonAssoc.label = ('NEWextractMe0Muons',)
NEWme0MuonTrackVMuonAssoc.muonTPSelector = NewMe0MuonTPSet
NEWme0MuonTrackVMuonAssoc.muonHistoParameters = me0MuonHistoParameters

# cosmics 2-leg reco
NEWtrkCosmicMuonTrackVSelMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtrkCosmicMuonTrackVSelMuonAssoc.associatormap = 'NEWtpToTkCosmicSelMuonAssociation'
NEWtrkCosmicMuonTrackVSelMuonAssoc.label = ('ctfWithMaterialTracksP5LHCNavigation',)
NEWtrkCosmicMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
NEWtrkCosmicMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
NEWtrkCosmicMuonTrackVSelMuonAssoc.BiDirectional_RecoToSim_association = False
NEWtrkCosmicMuonTrackVSelMuonAssoc.muonHistoParameters = trkCosmicMuonHistoParameters

NEWstaCosmicMuonTrackVSelMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaCosmicMuonTrackVSelMuonAssoc.associatormap = 'NEWtpToStaCosmicSelMuonAssociation'
NEWstaCosmicMuonTrackVSelMuonAssoc.label = ('cosmicMuons',)
NEWstaCosmicMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
NEWstaCosmicMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
NEWstaCosmicMuonTrackVSelMuonAssoc.BiDirectional_RecoToSim_association = False
NEWstaCosmicMuonTrackVSelMuonAssoc.muonHistoParameters = staCosmicMuonHistoParameters

NEWglbCosmicMuonTrackVSelMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWglbCosmicMuonTrackVSelMuonAssoc.associatormap = 'NEWtpToGlbCosmicSelMuonAssociation'
NEWglbCosmicMuonTrackVSelMuonAssoc.label = ('globalCosmicMuons',)
NEWglbCosmicMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
NEWglbCosmicMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
NEWglbCosmicMuonTrackVSelMuonAssoc.BiDirectional_RecoToSim_association = False
NEWglbCosmicMuonTrackVSelMuonAssoc.muonHistoParameters = glbCosmicMuonHistoParameters

# cosmics 1-leg reco
NEWtrkCosmic1LegMuonTrackVSelMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtrkCosmic1LegMuonTrackVSelMuonAssoc.associatormap = 'NEWtpToTkCosmic1LegSelMuonAssociation'
NEWtrkCosmic1LegMuonTrackVSelMuonAssoc.label = ('ctfWithMaterialTracksP5',)
NEWtrkCosmic1LegMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
NEWtrkCosmic1LegMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
NEWtrkCosmic1LegMuonTrackVSelMuonAssoc.muonHistoParameters = trkCosmic1LegMuonHistoParameters

NEWstaCosmic1LegMuonTrackVSelMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaCosmic1LegMuonTrackVSelMuonAssoc.associatormap = 'NEWtpToStaCosmic1LegSelMuonAssociation'
NEWstaCosmic1LegMuonTrackVSelMuonAssoc.label = ('cosmicMuons1Leg',)
NEWstaCosmic1LegMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
NEWstaCosmic1LegMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
NEWstaCosmic1LegMuonTrackVSelMuonAssoc.muonHistoParameters = staCosmic1LegMuonHistoParameters

NEWglbCosmic1LegMuonTrackVSelMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWglbCosmic1LegMuonTrackVSelMuonAssoc.associatormap = 'NEWtpToGlbCosmic1LegSelMuonAssociation'
NEWglbCosmic1LegMuonTrackVSelMuonAssoc.label = ('globalCosmicMuons1Leg',)
NEWglbCosmic1LegMuonTrackVSelMuonAssoc.parametersDefiner = cms.string('CosmicParametersDefinerForTP')
NEWglbCosmic1LegMuonTrackVSelMuonAssoc.muonTPSelector = cosmicMuonTPSet
NEWglbCosmic1LegMuonTrackVSelMuonAssoc.muonHistoParameters = glbCosmic1LegMuonHistoParameters

##################################################################################
# Muon validation sequences using MuonTrackValidator
#
NEWmuonValidation_seq = cms.Sequence(
    NEWprobeTracks_seq + NEWtpToTkMuonAssociation + NEWtrkProbeTrackVMuonAssoc
    +NEWtrackAssociatorByHits + NEWtpToTkmuTrackAssociation + NEWtrkMuonTrackVTrackAssoc
    +NEWseedsOfSTAmuons_seq + NEWtpToStaSeedAssociation + NEWstaSeedTrackVMuonAssoc
    +NEWtpToStaMuonAssociation + NEWstaMuonTrackVMuonAssoc
    +NEWtpToStaUpdMuonAssociation + NEWstaUpdMuonTrackVMuonAssoc
    +NEWtpToGlbMuonAssociation + NEWglbMuonTrackVMuonAssoc
)

NEWmuonValidation_reduced_seq = cms.Sequence(
    NEWprobeTracks_seq + NEWtpToTkMuonAssociation + NEWtrkProbeTrackVMuonAssoc
    +NEWtpToStaUpdMuonAssociation + NEWstaUpdMuonTrackVMuonAssoc
    +NEWtpToGlbMuonAssociation + NEWglbMuonTrackVMuonAssoc
    +NEWtpToDisplacedStaMuonAssociation + NEWdisplacedStaMuonTrackVMuonAssoc
    +NEWtpToDisplacedTrkMuonAssociation + NEWdisplacedTrackVMuonAssoc
    +NEWtpToDisplacedGlbMuonAssociation + NEWdisplacedGlbMuonTrackVMuonAssoc
)

NEWmuonValidationTEV_seq = cms.Sequence(
    NEWtpToTevFirstMuonAssociation + NEWtevMuonFirstTrackVMuonAssoc
    +NEWtpToTevPickyMuonAssociation + NEWtevMuonPickyTrackVMuonAssoc
    +NEWtpToTevDytMuonAssociation + NEWtevMuonDytTrackVMuonAssoc
)

NEWmuonValidationRefit_seq = cms.Sequence(
    NEWtpToStaRefitMuonAssociation + NEWstaRefitMuonTrackVMuonAssoc
    +NEWtpToStaRefitUpdMuonAssociation + NEWstaRefitUpdMuonTrackVMuonAssoc
)

NEWmuonValidationDisplaced_seq = cms.Sequence(
    NEWseedsOfDisplacedSTAmuons_seq + NEWtpToDisplacedStaSeedAssociation + NEWdisplacedStaSeedTrackVMuonAssoc
    +NEWtpToDisplacedStaMuonAssociation + NEWdisplacedStaMuonTrackVMuonAssoc
    +NEWtpToDisplacedTrkMuonAssociation + NEWdisplacedTrackVMuonAssoc
    +NEWtpToDisplacedGlbMuonAssociation + NEWdisplacedGlbMuonTrackVMuonAssoc
)

NEWmuonValidationCosmic_seq = cms.Sequence(
    NEWtpToTkCosmicSelMuonAssociation + NEWtrkCosmicMuonTrackVSelMuonAssoc
    +NEWtpToTkCosmic1LegSelMuonAssociation + NEWtrkCosmic1LegMuonTrackVSelMuonAssoc
    +NEWtpToStaCosmicSelMuonAssociation + NEWstaCosmicMuonTrackVSelMuonAssoc
    +NEWtpToStaCosmic1LegSelMuonAssociation + NEWstaCosmic1LegMuonTrackVSelMuonAssoc
    +NEWtpToGlbCosmicSelMuonAssociation + NEWglbCosmicMuonTrackVSelMuonAssoc
    +NEWtpToGlbCosmic1LegSelMuonAssociation + NEWglbCosmic1LegMuonTrackVSelMuonAssoc
)

NEWgemMuonValidation = cms.Sequence(NEWextractGemMuonsTracks_seq + NEWtpToGEMMuonMuonAssociation + NEWgemMuonTrackVMuonAssoc)
NEWme0MuonValidation = cms.Sequence(NEWextractMe0MuonsTracks_seq + NEWtpToME0MuonMuonAssociation + NEWme0MuonTrackVMuonAssoc)

##########################################################################
# The full offline muon validation sequence
#
NEWrecoMuonValidation = cms.Sequence(
    NEWmuonValidation_seq + NEWmuonValidationTEV_seq + NEWmuonValidationRefit_seq + NEWmuonValidationDisplaced_seq + NEWmuonValidationRMV_seq
    )

# no displaced muons in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    NEWrecoMuonValidation = cms.Sequence(NEWmuonValidation_seq + NEWmuonValidationTEV_seq + NEWmuonValidationRefit_seq)

# sequence for cosmic muons
NEWrecoCosmicMuonValidation = cms.Sequence(
    NEWmuonValidationCosmic_seq
    )

# sequences for muon upgrades
#
NEW_run3_muonValidation = NEWmuonValidation_seq.copy() #For full validation
NEW_run3_muonValidation = NEWmuonValidation_reduced_seq.copy()
NEW_run3_muonValidation += NEWgemMuonValidation

NEW_phase2_muonValidation = NEW_run3_muonValidation.copy()
NEW_phase2_muonValidation += NEWme0MuonValidation

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( NEWmuonValidation_seq, NEW_run3_muonValidation ) #For full validation
run3_GEM.toReplaceWith( NEWrecoMuonValidation, NEW_run3_muonValidation )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( NEWmuonValidation_seq, NEW_phase2_muonValidation ) #For full validation
phase2_muon.toReplaceWith( NEWrecoMuonValidation, NEW_phase2_muonValidation )
