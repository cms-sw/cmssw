import FWCore.ParameterSet.Config as cms
from Validation.HGCalValidation.HGCalValidator_cfi import hgcalValidator

layerClusterSimClusterAssociation = cms.EDProducer("LCToSCAssociatorEDProducer",
    associator = cms.InputTag('scAssocByEnergyScoreProducer'),
    label_scl = hgcalValidator.label_scl,
    label_lcl = hgcalValidator.label_lcl
)
