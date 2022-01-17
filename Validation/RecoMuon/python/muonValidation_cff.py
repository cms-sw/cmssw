#
# Production configuration for FullSim: muon track validation using MuonAssociatorByHits
#
import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.track_selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *

from Validation.RecoMuon.RecoMuonValidator_cff import *

import Validation.RecoMuon.MuonTrackValidator_cfi
MTV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone(
# DEFAULTS ###################################
#    label_tp = "mix:MergedTrackTruth",
#    label_tp_refvector = False,
#    muonTPSelector = cms.PSet(muonTPSet),
##############################################
    label_tp = ("TPmu"),
    label_tp_refvector = True
)
MTV.muonTPSelector.src = ("TPmu")
##############################################

trkMuonTrackVTrackAssoc = MTV.clone(
    associatormap = 'tpToTkmuTrackAssociation',
    associators = ('trackAssociatorByHits',),
    #label = ('generalTracks',),
    label = ('probeTracks',),
    label_tp = ("TPtrack"),
    muonHistoParameters = trkMuonHistoParameters
)
trkMuonTrackVTrackAssoc.muonTPSelector.src = ("TPtrack")
# MuonAssociatorByHits used for all track collections

trkProbeTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToTkMuonAssociation',
    #label = ('generalTracks',),
    label = ('probeTracks',),
    label_tp = ("TPtrack"),
    muonHistoParameters = trkMuonHistoParameters
)
trkProbeTrackVMuonAssoc.muonTPSelector.src = ("TPtrack")
staSeedTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToStaSeedAssociation',
    label = ('seedsOfSTAmuons',),
    muonHistoParameters = staSeedMuonHistoParameters
)
staMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToStaMuonAssociation',
    label = ('standAloneMuons',),
    muonHistoParameters = staMuonHistoParameters
)
staUpdMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToStaUpdMuonAssociation',
    label = ('standAloneMuons:UpdatedAtVtx',),
    muonHistoParameters = staUpdMuonHistoParameters
)
glbMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToGlbMuonAssociation',
    label = ('globalMuons',),
    muonHistoParameters = glbMuonHistoParameters
)
staRefitMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToStaRefitMuonAssociation',
    label = ('refittedStandAloneMuons',),
    muonHistoParameters = staMuonHistoParameters
)
staRefitUpdMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToStaRefitUpdMuonAssociation',
    label = ('refittedStandAloneMuons:UpdatedAtVtx',),
    muonHistoParameters = staUpdMuonHistoParameters
)
displacedTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToDisplacedTrkMuonAssociation',
    label = ('displacedTracks',),
    label_tp = ("TPtrack"),
    muonTPSelector = displacedMuonTPSet,
    muonHistoParameters = displacedTrkMuonHistoParameters
)
displacedTrackVMuonAssoc.muonTPSelector.src = ("TPtrack")

displacedStaSeedTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToDisplacedStaSeedAssociation',
    label = ('seedsOfDisplacedSTAmuons',),
    muonTPSelector = displacedMuonTPSet,
    muonHistoParameters = displacedStaSeedMuonHistoParameters
)
displacedStaSeedTrackVMuonAssoc.muonTPSelector.src = ("TPmu")

displacedStaMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToDisplacedStaMuonAssociation',
    label = ('displacedStandAloneMuons',),
    muonTPSelector = displacedMuonTPSet,
    muonHistoParameters = displacedStaMuonHistoParameters
)
displacedStaMuonTrackVMuonAssoc.muonTPSelector.src = ("TPmu")

displacedGlbMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToDisplacedGlbMuonAssociation',
    label = ('displacedGlobalMuons',),
    muonTPSelector = displacedMuonTPSet,
    muonHistoParameters = displacedGlbMuonHistoParameters
)
displacedGlbMuonTrackVMuonAssoc.muonTPSelector.src = ("TPmu")

tevMuonFirstTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToTevFirstMuonAssociation',
    label = ('tevMuons:firstHit',),
    muonHistoParameters = glbMuonHistoParameters
)
tevMuonPickyTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToTevPickyMuonAssociation',
    label = ('tevMuons:picky',),
    muonHistoParameters = glbMuonHistoParameters
)
tevMuonDytTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToTevDytMuonAssociation',
    label = ('tevMuons:dyt',),
    muonHistoParameters = glbMuonHistoParameters
)
tunepMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToTunePMuonAssociation',
    label = ('tunepMuonTracks',),
    muonHistoParameters = glbMuonHistoParameters
)
pfMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToPFMuonAssociation',
    label = ('pfMuonTracks',),
    label_tp = ("TPpfmu"),
    muonHistoParameters = glbMuonHistoParameters
)
pfMuonTrackVMuonAssoc.muonTPSelector.src = ("TPpfmu")

recomuMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpTorecoMuonMuonAssociation',
    label = ('recoMuonTracks',),
    muonHistoParameters = glbMuonHistoParameters
)
gemMuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToGEMMuonMuonAssociation',
    label = ('extractGemMuons',),
    muonHistoParameters = gemMuonHistoParameters
)
me0MuonTrackVMuonAssoc = MTV.clone(
    associatormap = 'tpToME0MuonMuonAssociation',
    label = ('extractMe0Muons',),
    muonTPSelector = me0MuonTPSet,
    muonHistoParameters = me0MuonHistoParameters
)
me0MuonTrackVMuonAssoc.muonTPSelector.src = ("TPmu")

MTVcosmic = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone(
# DEFAULTS ###################################
#    label_tp = "mix:MergedTrackTruth",
#    label_tp_refvector = False,
##############################################
    parametersDefiner = 'CosmicParametersDefinerForTP',
    muonTPSelector = cosmicMuonTPSet
)
##############################################

# cosmics 2-leg reco
trkCosmicMuonTrackVSelMuonAssoc = MTVcosmic.clone(
    associatormap = 'tpToTkCosmicSelMuonAssociation',
    label = ('ctfWithMaterialTracksP5LHCNavigation',),
    BiDirectional_RecoToSim_association = False,
    muonHistoParameters = trkCosmicMuonHistoParameters
)
staCosmicMuonTrackVSelMuonAssoc = MTVcosmic.clone(
    associatormap = 'tpToStaCosmicSelMuonAssociation',
    label = ('cosmicMuons',),
    BiDirectional_RecoToSim_association = False,
    muonHistoParameters = staCosmicMuonHistoParameters
)
glbCosmicMuonTrackVSelMuonAssoc = MTVcosmic.clone(
    associatormap = 'tpToGlbCosmicSelMuonAssociation',
    label = ('globalCosmicMuons',),
    BiDirectional_RecoToSim_association = False,
    muonHistoParameters = glbCosmicMuonHistoParameters
)
# cosmics 1-leg reco
trkCosmic1LegMuonTrackVSelMuonAssoc = MTVcosmic.clone(
    associatormap = 'tpToTkCosmic1LegSelMuonAssociation',
    label = ('ctfWithMaterialTracksP5',),
    muonHistoParameters = trkCosmic1LegMuonHistoParameters
)
staCosmic1LegMuonTrackVSelMuonAssoc = MTVcosmic.clone(
    associatormap = 'tpToStaCosmic1LegSelMuonAssociation',
    label = ('cosmicMuons1Leg',),
    muonHistoParameters = staCosmic1LegMuonHistoParameters
)
glbCosmic1LegMuonTrackVSelMuonAssoc = MTVcosmic.clone(
    associatormap = 'tpToGlbCosmic1LegSelMuonAssociation',
    label = ('globalCosmicMuons1Leg',),
    muonHistoParameters = glbCosmic1LegMuonHistoParameters
)

##########################################################################                                                        
### Customization for Phase II samples                                                                                           
###

trkMuonTrackVTrackAssoc_phase2 = trkMuonTrackVTrackAssoc.clone(                                                                  
    muonHistoParameters = trkMuonHistoParameters_phase2                                                
)
trkProbeTrackVMuonAssoc_phase2 = trkProbeTrackVMuonAssoc.clone(
    muonHistoParameters = trkMuonHistoParameters_phase2                                                
)
staSeedTrackVMuonAssoc_phase2 = staSeedTrackVMuonAssoc.clone(
    muonHistoParameters = staSeedMuonHistoParameters                                                    
)
staMuonTrackVMuonAssoc_phase2 = staMuonTrackVMuonAssoc.clone(
    muonHistoParameters = staMuonHistoParameters_phase2                                                 
)
staUpdMuonTrackVMuonAssoc_phase2 = staUpdMuonTrackVMuonAssoc.clone(
    muonHistoParameters = staUpdMuonHistoParameters_phase2                                          
)
glbMuonTrackVMuonAssoc_phase2 = glbMuonTrackVMuonAssoc.clone(
    muonHistoParameters = glbMuonHistoParameters_phase2                                                 
)
pfMuonTrackVMuonAssoc_phase2 = pfMuonTrackVMuonAssoc.clone(                                                                      
    muonHistoParameters = glbMuonHistoParameters_phase2                                                  
)
recomuMuonTrackVMuonAssoc_phase2 = recomuMuonTrackVMuonAssoc.clone(
    muonHistoParameters = recoMuonHistoParameters_phase2      
)
tunepMuonTrackVMuonAssoc_phase2 = tunepMuonTrackVMuonAssoc.clone(
    muonHistoParameters = glbMuonHistoParameters_phase2      
)
displacedStaMuonTrackVMuonAssoc_phase2 = displacedStaMuonTrackVMuonAssoc.clone(
    muonHistoParameters = displacedStaMuonHistoParameters_phase2   
)
displacedGlbMuonTrackVMuonAssoc_phase2 = displacedGlbMuonTrackVMuonAssoc.clone(
    muonHistoParameters = displacedGlbMuonHistoParameters_phase2               
)
displacedTrackVMuonAssoc_phase2 = displacedTrackVMuonAssoc.clone(
    muonHistoParameters = displacedTrkMuonHistoParameters_phase2   
)
gemMuonTrackVMuonAssoc_phase2 = gemMuonTrackVMuonAssoc.clone(
    muonHistoParameters = gemMuonHistoParameters_phase2 
)

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

muonValidation_noTABH_seq = cms.Sequence(
    probeTracks_seq + tpToTkMuonAssociation + trkProbeTrackVMuonAssoc
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

recoMuonValidation_reduced_seq = cms.Sequence(
    probeTracks_seq + tpToTkMuonAssociation + trkProbeTrackVMuonAssoc_phase2
    +tpToStaUpdMuonAssociation + staUpdMuonTrackVMuonAssoc_phase2
    +tpToGlbMuonAssociation + glbMuonTrackVMuonAssoc_phase2
    +tunepMuonTracks_seq + tpToTunePMuonAssociation + tunepMuonTrackVMuonAssoc_phase2
    +pfMuonTracks_seq + tpToPFMuonAssociation + pfMuonTrackVMuonAssoc_phase2
    +recoMuonTracks_seq + tpTorecoMuonMuonAssociation + recomuMuonTrackVMuonAssoc_phase2
    +tpToDisplacedStaMuonAssociation + displacedStaMuonTrackVMuonAssoc_phase2
    +tpToDisplacedTrkMuonAssociation + displacedTrackVMuonAssoc_phase2
    +tpToDisplacedGlbMuonAssociation + displacedGlbMuonTrackVMuonAssoc_phase2
)

muonValidationCosmic_seq = cms.Sequence(
    tpToTkCosmicSelMuonAssociation + trkCosmicMuonTrackVSelMuonAssoc
    +tpToTkCosmic1LegSelMuonAssociation + trkCosmic1LegMuonTrackVSelMuonAssoc
    +tpToStaCosmicSelMuonAssociation + staCosmicMuonTrackVSelMuonAssoc
    +tpToStaCosmic1LegSelMuonAssociation + staCosmic1LegMuonTrackVSelMuonAssoc
    +tpToGlbCosmicSelMuonAssociation + glbCosmicMuonTrackVSelMuonAssoc
    +tpToGlbCosmic1LegSelMuonAssociation + glbCosmic1LegMuonTrackVSelMuonAssoc
)

gemMuonValidation = cms.Sequence(extractGemMuonsTracks_seq + tpToGEMMuonMuonAssociation + gemMuonTrackVMuonAssoc)
me0MuonValidation = cms.Sequence(extractMe0MuonsTracks_seq + tpToME0MuonMuonAssociation + me0MuonTrackVMuonAssoc)

gemMuonValidation_phase2 = cms.Sequence(extractGemMuonsTracks_seq + tpToGEMMuonMuonAssociation + gemMuonTrackVMuonAssoc_phase2) 

##########################################################################
# The full offline muon validation sequence
#
recoMuonValidation = cms.Sequence( TPtrack_seq + TPmu_seq + TPpfmu_seq +
    muonValidation_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationDisplaced_seq + muonValidationRMV_seq
    )

# optionally omit TABH
recoMuonValidation_noTABH = cms.Sequence( TPtrack_seq + TPmu_seq + TPpfmu_seq +
    muonValidation_noTABH_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationDisplaced_seq + muonValidationRMV_seq
    )

# ... and also displaced muons
recoMuonValidation_noTABH_noDisplaced = cms.Sequence( TPtrack_seq + TPmu_seq + TPpfmu_seq +
    muonValidation_noTABH_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationRMV_seq
    )

# no displaced muons in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(recoMuonValidation, cms.Sequence(cms.SequencePlaceholder("TPtrack") + cms.SequencePlaceholder("TPmu") + cms.SequencePlaceholder("TPpfmu") + muonValidation_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationRMV_seq))

# sequence for cosmic muons
recoCosmicMuonValidation = cms.Sequence(
    muonValidationCosmic_seq
    )

# sequences for muon upgrades
#
_run3_muonValidation = recoMuonValidation.copy()
_run3_muonValidation += gemMuonValidation

_phase2_muonValidation = cms.Sequence(TPtrack_seq + TPmu_seq + TPpfmu_seq + recoMuonValidation_reduced_seq)
_phase2_muonValidation += gemMuonValidation_phase2
_phase2_muonValidation += me0MuonValidation

_phase2_ge0_muonValidation =  cms.Sequence(TPtrack_seq + TPmu_seq + TPpfmu_seq + recoMuonValidation_reduced_seq)
_phase2_ge0_muonValidation += gemMuonValidation_phase2

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( recoMuonValidation, _run3_muonValidation )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith( recoMuonValidation, _phase2_muonValidation )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toReplaceWith( recoMuonValidation, _phase2_ge0_muonValidation )

