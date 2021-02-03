import FWCore.ParameterSet.Config as cms
from Validation.HGCalValidation.HGCalValidator_cfi import hgcalValidator

layerClusterCaloParticleAssociation = cms.EDProducer("LCToCPAssociatorEDProducer",
    associator = cms.InputTag('lcAssocByEnergyScoreProducer'),
    label_cp = hgcalValidator.label_cp_effic,
    label_lc = hgcalValidator.label_lcl
)
