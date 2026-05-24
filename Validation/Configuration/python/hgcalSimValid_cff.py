import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cff import scAssocByEnergyScoreProducer, layerClusterSimClusterAssociationProducer, layerClusterBoundaryTrackSimClusterAssociationProducer, layerClusterMergedSimClusterAssociationProducer, layerClusterCaloParticleSimClusterAssociationProducer
from SimCalorimetry.HGCalAssociatorProducers.LCToSCAssociation_cff import layerClusterSimClusterAssociationProducerHFNose, layerClusterCaloParticleSimClusterAssociationProducerHFNose
# from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationLinkingSuperclustering, tracksterSimTracksterAssociationPRSuperclustering #, tracksterSimTracksterAssociationLinkingbyCLUE3DEM, tracksterSimTracksterAssociationLinkingbyCLUE3DHAD, tracksterSimTracksterAssociationPRbyCLUE3DEM, tracksterSimTracksterAssociationPRbyCLUE3DHAD
from RecoHGCal.TICL.mergedTrackstersProducer_cfi import mergedTrackstersProducer as _mergedTrackstersProducer
from SimCalorimetry.HGCalAssociatorProducers.SimTauProducer_cfi import *


# FP 07/2024: new associators:
from SimCalorimetry.HGCalAssociatorProducers.LCToTSAssociator_cfi import allLayerClusterToTracksterAssociations
from SimCalorimetry.HGCalAssociatorProducers.HitToTracksterAssociation_cfi import allHitToTracksterAssociations
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import  allTrackstersToSimTrackstersAssociationsByLCs
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociationByHits_cfi import allTrackstersToSimTrackstersAssociationsByHits
from SimCalorimetry.HGCalAssociatorProducers.HitToSimClusterAssociation_cff import hitToLegacySimClusterAssociator, hitToBoundarySimClusterAssociator, hitToMergedSimClusterAssociator, hitToCPSimClusterAssociator
from SimCalorimetry.HGCalAssociatorProducers.SimClusterToCaloParticleAssociation_cfi import SimClusterToCaloParticleAssociation


from Validation.HGCalValidation.simhitValidation_cff    import *
from Validation.HGCalValidation.digiValidation_cff      import *
from Validation.HGCalValidation.rechitValidation_cff    import *
from Validation.HGCalValidation.hgcalHitValidation_cff  import *
from RecoHGCal.TICL.SimTracksters_cff import *


from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
from Validation.RecoParticleFlow.PFJetValidation_cff import pfJetValidation1 as _hgcalPFJetValidation

from Validation.HGCalValidation.ticlPFValidation_cfi import ticlPFValidation
hgcalTiclPFValidation = cms.Sequence(ticlPFValidation)

from Validation.HGCalValidation.ticlTrackstersEdgesValidation_cfi import ticlTrackstersEdgesValidation
hgcalTiclTrackstersEdgesValidationSequence = cms.Sequence(ticlTrackstersEdgesValidation)

from Validation.HGCalValidation.ticlSuperclusterValidation_cff import *

hgcalValidatorSequence = cms.Sequence(hgcalValidator)
hgcalPFJetValidation = _hgcalPFJetValidation.clone(BenchmarkLabel = 'PFJetValidation/HGCAlCompWithGenJet',
    VariablePtBins=[10., 30., 80., 120., 250., 600.],
    DeltaPtOvPtHistoParameter = dict(EROn=True,EREtaMax=3.0, EREtaMin=1.6, slicingOn=True))

hgcalAssociators = cms.Task(scAssocByEnergyScoreProducer,
                            layerClusterSimClusterAssociationProducer, layerClusterBoundaryTrackSimClusterAssociationProducer, layerClusterMergedSimClusterAssociationProducer, layerClusterCaloParticleSimClusterAssociationProducer,
                            SimTauProducer,
                            # FP 07/2024 new associators:
                            # layerClusterToCLUE3DTracksterAssociation, layerClusterToTracksterMergeAssociation,
                            # layerClusterToSimTracksterAssociation, layerClusterToSimTracksterFromCPsAssociation,
                            allLayerClusterToTracksterAssociations, allHitToTracksterAssociations, allTrackstersToSimTrackstersAssociationsByLCs, allTrackstersToSimTrackstersAssociationsByHits,
                            # hitToTrackstersAssociationLinking, hitToTrackstersAssociationPR,
                            # hitToSimTracksterAssociation, hitToSimTracksterFromCPsAssociation,
                            hitToLegacySimClusterAssociator, hitToBoundarySimClusterAssociator, hitToMergedSimClusterAssociator, hitToCPSimClusterAssociator,
                            SimClusterToCaloParticleAssociation, 
                            )

from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf


hgcalValidation = cms.Sequence(hgcalSimHitValidationEE
                               + hgcalSimHitValidationHEF
                               + hgcalSimHitValidationHEB
                               + hgcalDigiValidationEE
                               + hgcalDigiValidationHEF
                               + hgcalDigiValidationHEB
                               + hgcalRecHitValidationEE
                               + hgcalRecHitValidationHEF
                               + hgcalRecHitValidationHEB
                               + hgcalHitValidationSequence
                               + hgcalValidatorSequence
                               + hgcalTiclPFValidation
                               + ticlSuperclusterValidation
                               #Currently commented out until trackster edges are saved
#                               + hgcalTiclTrackstersEdgesValidationSequence
                               + hgcalPFJetValidation)

_hfnose_hgcalAssociatorsTask = hgcalAssociators.copy()
_hfnose_hgcalAssociatorsTask.add(layerClusterCaloParticleSimClusterAssociationProducerHFNose, layerClusterSimClusterAssociationProducerHFNose)
