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

#
# MuonAssociatorByHits used for all track collections
#
NEWtrkProbeTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtrkProbeTrackVMuonAssoc.associatormap = 'NEWtpToTkMuonAssociation'
#trkProbeTrackVMuonAssoc.label = ('generalTracks',)
NEWtrkProbeTrackVMuonAssoc.label = ('probeTracks',)
NEWtrkProbeTrackVMuonAssoc.muonHistoParameters = trkMuonHistoParameters

# quickTrackAssociatorByHits on probeTracks used as monitor wrt MuonAssociatorByHits
#
NEWtrkMuonTrackVTrackAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWtrkMuonTrackVTrackAssoc.associatormap = 'tpToTkmuTrackAssociation'
NEWtrkMuonTrackVTrackAssoc.associators = ('trackAssociatorByHits',)
#trkMuonTrackVTrackAssoc.label = ('generalTracks',)
NEWtrkMuonTrackVTrackAssoc.label = ('probeTracks',)
NEWtrkMuonTrackVTrackAssoc.muonHistoParameters = trkMuonHistoParameters

NEWstaSeedTrackVMuonAssoc = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWstaSeedTrackVMuonAssoc.associatormap = 'NEWtpToStaSeedAssociation'
NEWstaSeedTrackVMuonAssoc.label = ('seedsOfSTAmuons',)
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
NEWdisplacedStaSeedTrackVMuonAssoc.label = ('seedsOfDisplacedSTAmuons',)
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
    probeTracks_seq + NEWtpToTkMuonAssociation + NEWtrkProbeTrackVMuonAssoc
    +trackAssociatorByHits + tpToTkmuTrackAssociation + NEWtrkMuonTrackVTrackAssoc
    +seedsOfSTAmuons_seq + NEWtpToStaSeedAssociation + NEWstaSeedTrackVMuonAssoc
    +NEWtpToStaMuonAssociation + NEWstaMuonTrackVMuonAssoc
    +NEWtpToStaUpdMuonAssociation + NEWstaUpdMuonTrackVMuonAssoc
    + NEWtpToGlbMuonAssociation + NEWglbMuonTrackVMuonAssoc
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
    seedsOfDisplacedSTAmuons_seq + NEWtpToDisplacedStaSeedAssociation + NEWdisplacedStaSeedTrackVMuonAssoc
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

#####################################################################################
# Configurations for RecoMuonValidator
#
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from Validation.RecoMuon.RecoMuonValidator_cfi import *

#import SimGeneral.MixingModule.mixNoPU_cfi
from SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi import *
from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters

#tracker
muonAssociatorByHitsNoSimHitsHelperTrk = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperTrk.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperTrk.UseMuon  = False
recoMuonVMuAssoc_trk = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_trk.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Trk'
recoMuonVMuAssoc_trk.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTrk'
recoMuonVMuAssoc_trk.trackType = 'inner'
recoMuonVMuAssoc_trk.selection = "isTrackerMuon"

#tracker and PF
muonAssociatorByHitsNoSimHitsHelperTrkPF = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperTrkPF.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperTrkPF.UseMuon  = False
recoMuonVMuAssoc_trkPF = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_trkPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_TrkPF'
recoMuonVMuAssoc_trkPF.usePFMuon = True
recoMuonVMuAssoc_trkPF.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTrkPF'
recoMuonVMuAssoc_trkPF.trackType = 'inner'
recoMuonVMuAssoc_trkPF.selection = "isTrackerMuon & isPFMuon"

#standalone
muonAssociatorByHitsNoSimHitsHelperStandalone = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperStandalone.UseTracker = False
muonAssociatorByHitsNoSimHitsHelperStandalone.UseMuon  = True
recoMuonVMuAssoc_sta = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_sta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Sta'
recoMuonVMuAssoc_sta.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalone'
recoMuonVMuAssoc_sta.trackType = 'outer'
recoMuonVMuAssoc_sta.selection = "isStandAloneMuon"

#seed of StandAlone
muonAssociatorByHitsNoSimHitsHelperSeedStandalone = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperSeedStandalone.UseTracker = False
muonAssociatorByHitsNoSimHitsHelperSeedStandalone.UseMuon  = True
recoMuonVMuAssoc_seedSta = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_seedSta.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_SeedSta'
recoMuonVMuAssoc_seedSta.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalone'
recoMuonVMuAssoc_seedSta.trackType = 'outer'
recoMuonVMuAssoc_seedSta.selection = ""

#standalone and PF
muonAssociatorByHitsNoSimHitsHelperStandalonePF = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperStandalonePF.UseTracker = False
muonAssociatorByHitsNoSimHitsHelperStandalonePF.UseMuon  = True
recoMuonVMuAssoc_staPF = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_staPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_StaPF'
recoMuonVMuAssoc_staPF.usePFMuon = True
recoMuonVMuAssoc_staPF.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperStandalonePF'
recoMuonVMuAssoc_staPF.trackType = 'outer'
recoMuonVMuAssoc_staPF.selection = "isStandAloneMuon & isPFMuon"

#global
muonAssociatorByHitsNoSimHitsHelperGlobal = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperGlobal.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperGlobal.UseMuon  = True
recoMuonVMuAssoc_glb = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_glb.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Glb'
recoMuonVMuAssoc_glb.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperGlobal'
recoMuonVMuAssoc_glb.trackType = 'global'
recoMuonVMuAssoc_glb.selection = "isGlobalMuon"

#global and PF
muonAssociatorByHitsNoSimHitsHelperGlobalPF = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperGlobalPF.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperGlobalPF.UseMuon  = True
recoMuonVMuAssoc_glbPF = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_glbPF.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_GlbPF'
recoMuonVMuAssoc_glbPF.usePFMuon = True
recoMuonVMuAssoc_glbPF.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperGlobalPF'
recoMuonVMuAssoc_glbPF.trackType = 'global'
recoMuonVMuAssoc_glbPF.selection = "isGlobalMuon & isPFMuon"

#tight
muonAssociatorByHitsNoSimHitsHelperTight = SimMuon.MCTruth.muonAssociatorByHitsNoSimHitsHelper_cfi.muonAssociatorByHitsNoSimHitsHelper.clone()
muonAssociatorByHitsNoSimHitsHelperTight.UseTracker = True
muonAssociatorByHitsNoSimHitsHelperTight.UseMuon  = True
recoMuonVMuAssoc_tgt = Validation.RecoMuon.RecoMuonValidator_cfi.recoMuonValidator.clone()
recoMuonVMuAssoc_tgt.subDir = 'Muons/RecoMuonV/RecoMuon_MuonAssoc_Tgt'
recoMuonVMuAssoc_tgt.muAssocLabel = 'muonAssociatorByHitsNoSimHitsHelperTight'
recoMuonVMuAssoc_tgt.trackType = 'global'
recoMuonVMuAssoc_tgt.selection = 'isGlobalMuon'
recoMuonVMuAssoc_tgt.wantTightMuon = True
recoMuonVMuAssoc_tgt.beamSpot = 'offlineBeamSpot'
recoMuonVMuAssoc_tgt.primaryVertex = 'offlinePrimaryVertices'

##########################################################################
# Muon validation sequence using RecoMuonValidator
#
muonValidationRMV_seq = cms.Sequence(
    muonAssociatorByHitsNoSimHitsHelperTrk +recoMuonVMuAssoc_trk
    +muonAssociatorByHitsNoSimHitsHelperStandalone +recoMuonVMuAssoc_sta
    +muonAssociatorByHitsNoSimHitsHelperGlobal +recoMuonVMuAssoc_glb
    +muonAssociatorByHitsNoSimHitsHelperTight +recoMuonVMuAssoc_tgt
    # 
    #    +muonAssociatorByHitsNoSimHitsHelperTrkPF +recoMuonVMuAssoc_trkPF
    #    +muonAssociatorByHitsNoSimHitsHelperStandalonePF +recoMuonVMuAssoc_staPF
    #    +muonAssociatorByHitsNoSimHitsHelperGlobalPF +recoMuonVMuAssoc_glbPF
    )

##########################################################################
# The full offline muon validation sequence
#
NEWrecoMuonValidation = cms.Sequence(
    NEWmuonValidation_seq + NEWmuonValidationTEV_seq + NEWmuonValidationRefit_seq + NEWmuonValidationDisplaced_seq
    + muonValidationRMV_seq
    )

from Configuration.StandardSequences.Eras import eras
# no displaces or SET muons in fastsim
if eras.fastSim.isChosen():
    NEWrecoMuonValidation = cms.Sequence(NEWmuonValidation_seq + NEWmuonValidationTEV_seq + NEWmuonValidationRefit_seq)

# sequence for cosmic muons
NEWrecoCosmicMuonValidation = cms.Sequence(
    NEWmuonValidationCosmic_seq
    )
