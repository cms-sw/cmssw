import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import lcAssocByEnergyScoreProducer, scAssocByEnergyScoreProducer
from SimDataFormats.Associations.LCToSCAsssociation_cfi import layerClusterSimClusterAsssociation as layerClusterSimClusterAsssociationProducer

from Validation.HGCalValidation.simhitValidation_cff    import *
from Validation.HGCalValidation.digiValidation_cff      import *
from Validation.HGCalValidation.rechitValidation_cff    import *
from Validation.HGCalValidation.hgcalHitValidation_cfi  import *

from Validation.HGCalValidation.HGCalValidator_cfi import hgcalValidator
from Validation.RecoParticleFlow.PFJetValidation_cff import pfJetValidation1 as _hgcalPFJetValidation

from Validation.HGCalValidation.ticlPFValidation_cfi import ticlPFValidation
hgcalTiclPFValidation = cms.Sequence(ticlPFValidation)

hgcalValidatorSequence = cms.Sequence(hgcalValidator)
hgcalPFJetValidation = _hgcalPFJetValidation.clone(BenchmarkLabel = 'PFJetValidation/HGCAlCompWithGenJet',
    VariablePtBins=[10., 30., 80., 120., 250., 600.],
    DeltaPtOvPtHistoParameter = dict(EROn=True,EREtaMax=3.0, EREtaMin=1.6, slicingOn=True))

scAssocByEnergyScoreProducerticlTrackstersTrkEM = scAssocByEnergyScoreProducer.clone(
    LayerClustersInputMask = 'ticlTrackstersTrkEM')
layerClusterSimClusterAsssociationTrkEM = layerClusterSimClusterAsssociationProducer.clone(
    associator = cms.InputTag('scAssocByEnergyScoreProducerticlTrackstersTrkEM')
)

scAssocByEnergyScoreProducerticlTrackstersEM = scAssocByEnergyScoreProducer.clone(
    LayerClustersInputMask = 'ticlTrackstersEM')
layerClusterSimClusterAsssociationEM = layerClusterSimClusterAsssociationProducer.clone(
    associator = cms.InputTag('scAssocByEnergyScoreProducerticlTrackstersEM')
)

scAssocByEnergyScoreProducerticlTrackstersTrk = scAssocByEnergyScoreProducer.clone(
    LayerClustersInputMask = 'ticlTrackstersTrk')
layerClusterSimClusterAsssociationTrk = layerClusterSimClusterAsssociationProducer.clone(
    associator = cms.InputTag('scAssocByEnergyScoreProducerticlTrackstersTrk')
)

scAssocByEnergyScoreProducerticlTrackstersHAD = scAssocByEnergyScoreProducer.clone(
    LayerClustersInputMask = 'ticlTrackstersHAD')
layerClusterSimClusterAsssociationHAD = layerClusterSimClusterAsssociationProducer.clone(
    associator = cms.InputTag('scAssocByEnergyScoreProducerticlTrackstersHAD')
)

hgcalAssociators = cms.Task(lcAssocByEnergyScoreProducer,
                            scAssocByEnergyScoreProducerticlTrackstersTrkEM, layerClusterSimClusterAsssociationTrkEM,
                            scAssocByEnergyScoreProducerticlTrackstersEM, layerClusterSimClusterAsssociationEM,
                            scAssocByEnergyScoreProducerticlTrackstersTrk, layerClusterSimClusterAsssociationTrk,
                            scAssocByEnergyScoreProducerticlTrackstersHAD, layerClusterSimClusterAsssociationHAD
                            )

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
                               + hgcalPFJetValidation)
