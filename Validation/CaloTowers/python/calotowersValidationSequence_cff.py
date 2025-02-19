import FWCore.ParameterSet.Config as cms

from Validation.CaloTowers.CaloTowersParam_cfi import *
import Validation.CaloTowers.CaloTowersParam_cfi
AllCaloTowersValidation = Validation.CaloTowers.CaloTowersParam_cfi.calotowersAnalyzer.clone()
calotowersValidationSequence = cms.Sequence(AllCaloTowersValidation)
