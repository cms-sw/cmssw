#
# Production configuration for FullSim: muon track validation using MuonAssociatorByHits
#
import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.track_selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *

from Validation.RecoMuon.RecoMuonValidator_cff import *
from Validation.RecoMuon.RecoDisplacedMuonValidator_cff import *

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
    associatormap = ('tpToTkmuTrackAssociation',),
    associators = ('trackAssociatorByHits',),
    #label = ('generalTracks',),
    label = ('probeTracks',),
    label_tp = ("TPtrack"),
    muonHistoParameters = (trkMuonHistoParameters,)
)
trkMuonTrackVTrackAssoc.muonTPSelector.src = ("TPtrack")
# MuonAssociatorByHits used for all track collections

trkProbeTrackVMuonAssoc = MTV.clone(
    associatormap = ('tpToTkMuonAssociation',),
    #label = ('generalTracks',),
    label = ('probeTracks',),
    label_tp = ("TPtrack"),
    muonHistoParameters = (trkMuonHistoParameters,)
)
trkProbeTrackVMuonAssoc.muonTPSelector.src = ("TPtrack")

# The Muon Multi Track Validator

muonMultiTrackValidator = MTV.clone(
    associatormap = (
        'tpToStaSeedAssociation',
        'tpToStaMuonAssociation',
        'tpToStaUpdMuonAssociation',
        'tpToGlbMuonAssociation',
        'tpTorecoMuonMuonAssociation'
    ),
    label = (
        'seedsOfSTAmuons',
        'standAloneMuons',
        'standAloneMuons:UpdatedAtVtx',
        'globalMuons',
        'recoMuonTracks'
    ),
    muonHistoParameters = (
        staSeedMuonHistoParameters,
        staMuonHistoParameters,
        staUpdMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters
    ),
    doSummaryPlots = True
)

muonMultiTrackValidator_phase2 = muonMultiTrackValidator.clone()
muonMultiTrackValidator_phase2.associatormap = ('tpToStaUpdMuonAssociation','tpToGlbMuonAssociation','tpTorecoMuonMuonAssociation')
muonMultiTrackValidator_phase2.label = ('standAloneMuons:UpdatedAtVtx','globalMuons','recoMuonTracks')
muonMultiTrackValidator_phase2.muonHistoParameters = (staUpdMuonHistoParameters,glbMuonHistoParameters,glbMuonHistoParameters)

# The Muon Multi Track Validator for refitted muons

muonMultiTrackValidatorRefit = MTV.clone(
    associatormap = (
        'tpToStaRefitMuonAssociation',
        'tpToStaRefitUpdMuonAssociation'
    ),
    label = (
        'refittedStandAloneMuons',
        'refittedStandAloneMuons:UpdatedAtVtx'
    ),
    muonHistoParameters = (
        staMuonHistoParameters,
        staUpdMuonHistoParameters
    )
)

# The Muon Multi Track Validator for displaced muons

displacedTrackVMuonAssoc = MTV.clone(
    associatormap = ('tpToDisplacedTrkMuonAssociation',),
    label = ('displacedTracks',),
    label_tp = ("TPtrack"),
    muonTPSelector = displacedMuonTPSet,
    muonHistoParameters = (displacedTrkMuonHistoParameters,)
)
displacedTrackVMuonAssoc.muonTPSelector.src = ("TPtrack")

displacedMuonMultiTrackValidator = MTV.clone(
    associatormap = (
        'tpToDisplacedStaSeedAssociation',
        'tpToDisplacedStaMuonAssociation',
        'tpToDisplacedGlbMuonAssociation',
    ),
    label = (
        'seedsOfDisplacedSTAmuons',
        'displacedStandAloneMuons',
        'displacedGlobalMuons'
    ),
    muonTPSelector = displacedMuonTPSet,
    muonHistoParameters = (
        displacedStaSeedMuonHistoParameters,
        displacedStaMuonHistoParameters,
        displacedGlbMuonHistoParameters
    )
)
displacedMuonMultiTrackValidator.muonTPSelector.src = ("TPmu")

displacedMuonMultiTrackValidator_phase2 = displacedMuonMultiTrackValidator.clone()
displacedMuonMultiTrackValidator_phase2.associatormap = ('tpToDisplacedStaMuonAssociation','tpToDisplacedGlbMuonAssociation')
displacedMuonMultiTrackValidator_phase2.label = ('displacedStandAloneMuons','displacedGlobalMuons')
displacedMuonMultiTrackValidator_phase2.muonHistoParameters = (displacedStaMuonHistoParameters,displacedGlbMuonHistoParameters)

# The Muon Multi Track Validator for TeV muons

tevMuonMultiTrackValidator = MTV.clone(
    associatormap = (
        'tpToTevFirstMuonAssociation',
        'tpToTevPickyMuonAssociation',
        'tpToTevDytMuonAssociation',
        'tpToTunePMuonAssociation'
    ),
    label = (
        'tevMuons:firstHit',
        'tevMuons:picky',
        'tevMuons:dyt',
        'tunepMuonTracks'
    ),
    muonHistoParameters = (
        glbMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters,
        glbMuonHistoParameters
    )
)

tevMuonMultiTrackValidator_phase2 = MTV.clone(
    associatormap = ('tpToTunePMuonAssociation',),
    label = ('tunepMuonTracks',),
    muonHistoParameters = (glbMuonHistoParameters,)
)

pfMuonTrackVMuonAssoc = MTV.clone(
    associatormap = ('tpToPFMuonAssociation',),
    label = ('pfMuonTracks',),
    label_tp = ("TPpfmu"),
    muonHistoParameters = (glbMuonHistoParameters,)
)
pfMuonTrackVMuonAssoc.muonTPSelector.src = ("TPpfmu")

gemMuonTrackVMuonAssoc = MTV.clone(
    associatormap = ('tpToGEMMuonMuonAssociation',),
    label = ('extractGemMuons',),
    muonHistoParameters = (gemMuonHistoParameters,)
)

me0MuonTrackVMuonAssoc = MTV.clone(
    associatormap = ('tpToME0MuonMuonAssociation',),
    label = ('extractMe0Muons',),
    muonTPSelector = me0MuonTPSet,
    muonHistoParameters = (me0MuonHistoParameters,)
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

cosmic2LegMuonMultiTrackValidator = MTVcosmic.clone(
    associatormap = (
        'tpToTkCosmicSelMuonAssociation',
        'tpToStaCosmicSelMuonAssociation',
        'tpToGlbCosmicSelMuonAssociation'
    ),
    label = (
        'ctfWithMaterialTracksP5LHCNavigation',
        'cosmicMuons',
        'globalCosmicMuons'
    ),
    BiDirectional_RecoToSim_association = False,
    muonHistoParameters = (
        trkCosmicMuonHistoParameters,
        staCosmicMuonHistoParameters,
        glbCosmicMuonHistoParameters
    )
)

cosmic1LegMuonMultiTrackValidator = MTVcosmic.clone(
    associatormap = (
        'tpToTkCosmic1LegSelMuonAssociation',
        'tpToStaCosmic1LegSelMuonAssociation',
        'tpToGlbCosmic1LegSelMuonAssociation',
    ),
    label = (
        'ctfWithMaterialTracksP5',
        'cosmicMuons1Leg',
        'globalCosmicMuons1Leg'
    ),
    muonHistoParameters = (
        trkCosmic1LegMuonHistoParameters,
        staCosmic1LegMuonHistoParameters,
        glbCosmic1LegMuonHistoParameters
    )
)

# Check that the associators and labels are consistent
# All MTV clones are DQMEDAnalyzers
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
# Access all the global variables
global_items = list(globals().items())
for _name, _obj in global_items:
    # Find all MTV clones
    if isinstance(_obj, DQMEDAnalyzer) and hasattr(_obj, 'label') and hasattr(_obj, 'associatormap') and hasattr(_obj, 'muonHistoParameters'):
        # Check that the size of the associators, lables and muonHistoParameters are the same
        if (len(_obj.label) != len(_obj.associatormap) or len(_obj.label) != len(_obj.muonHistoParameters)
            or len(_obj.associatormap) != len(_obj.muonHistoParameters)):
            raise RuntimeError(f"MuonTrackValidator -- {_name}: associatormap, label and muonHistoParameters must have the same length!")
        # Check that the trackCollection used in each associator corresponds to the validator's label
        for i in range(0, len(_obj.label)):
            # Dynamically import the associators module to have access to procModifiers changes
            associators_module = __import__('Validation.RecoMuon.associators_cff', globals(), locals(), ['associators'], 0)
            _assoc = getattr(associators_module, _obj.associatormap[i].value()) if isinstance(_obj.associatormap[i], cms.InputTag) else getattr(associators_module, _obj.associatormap[i])
            _label = _obj.label[i].value() if isinstance(_obj.label[i], cms.InputTag) else _obj.label[i]
            _tracksTag = _assoc.tracksTag.value() if hasattr(_assoc, 'tracksTag') else _assoc.label_tr.value()
            if _tracksTag != _label:
                raise RuntimeError(f"MuonTrackValidator -- {_name}: associatormap and label do not match for index {i}.\n"
                                   f"Associator's tracksTag: {_tracksTag}, collection label in the validator: {_label}.\n"
                                   "Make sure to have the correct ordering!")

##################################################################################
# Muon validation sequences using MuonTrackValidator
#
muonValidation_seq = cms.Sequence(muonAssociation_seq 
                                  +trkMuonTrackVTrackAssoc
                                  +trkProbeTrackVMuonAssoc 
                                  +muonMultiTrackValidator 
                                  +pfMuonTrackVMuonAssoc
)

muonValidationTEV_seq = cms.Sequence(muonAssociationTEV_seq 
                                     +tevMuonMultiTrackValidator
)

muonValidationRefit_seq = cms.Sequence(muonAssociationRefit_seq 
                                       +muonMultiTrackValidatorRefit
)

muonValidationDisplaced_seq = cms.Sequence(muonAssociationDisplaced_seq 
                                           +displacedTrackVMuonAssoc 
                                           +displacedMuonMultiTrackValidator
)

muonValidationCosmic_seq = cms.Sequence(muonAssociationCosmic_seq 
                                        +cosmic2LegMuonMultiTrackValidator 
                                        +cosmic1LegMuonMultiTrackValidator
)

gemMuonValidation = cms.Sequence(extractGemMuonsTracks_seq 
                                 +tpToGEMMuonMuonAssociation 
                                 +gemMuonTrackVMuonAssoc
)

me0MuonValidation = cms.Sequence(extractMe0MuonsTracks_seq 
                                 +tpToME0MuonMuonAssociation 
                                 +me0MuonTrackVMuonAssoc
)

##########################################################################
# The full offline muon validation sequence
#
recoMuonValidation = cms.Sequence(TPtrack_seq 
                                  +TPmu_seq 
                                  +TPpfmu_seq 
                                  +muonValidation_seq 
                                  +muonValidationTEV_seq 
                                  +muonValidationRefit_seq 
                                  +muonValidationDisplaced_seq 
                                  +muonValidationRMV_seq 
                                  +muonValidationRDMV_seq
)

# no displaced muons in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(recoMuonValidation, cms.Sequence(cms.SequencePlaceholder("TPtrack") + cms.SequencePlaceholder("TPmu") + cms.SequencePlaceholder("TPpfmu") + muonValidation_seq + muonValidationTEV_seq + muonValidationRefit_seq + muonValidationRMV_seq))

# sequence for cosmic muons
recoCosmicMuonValidation = cms.Sequence(muonValidationCosmic_seq)

# Phase 2 reduced sequence
recoMuonValidation_reduced_seq = cms.Sequence(muonAssociationReduced_seq 
                                              +trkProbeTrackVMuonAssoc
                                              +muonMultiTrackValidator_phase2
                                              +tevMuonMultiTrackValidator_phase2
                                              +pfMuonTrackVMuonAssoc 
                                              +displacedTrackVMuonAssoc
                                              +displacedMuonMultiTrackValidator_phase2
)

#
# sequences for muon upgrades
#
_run3_muonValidation = recoMuonValidation.copy()
_run3_muonValidation += gemMuonValidation

_phase2_muonValidation = cms.Sequence(TPtrack_seq 
                                      +TPmu_seq 
                                      +TPpfmu_seq 
                                      +recoMuonValidation_reduced_seq
                                      +gemMuonValidation
                                      +me0MuonValidation
)

_phase2_ge0_muonValidation =  cms.Sequence(TPtrack_seq 
                                           +TPmu_seq 
                                           +TPpfmu_seq 
                                           +recoMuonValidation_reduced_seq
                                           +gemMuonValidation
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith(recoMuonValidation, _run3_muonValidation)
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(recoMuonValidation, _phase2_muonValidation)
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toReplaceWith(recoMuonValidation, _phase2_ge0_muonValidation)
