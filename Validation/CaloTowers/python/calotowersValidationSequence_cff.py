import FWCore.ParameterSet.Config as cms

from Validation.CaloTowers.CaloTowersParam_cfi import *
import Validation.CaloTowers.CaloTowersParam_cfi
HBCaloTowersValidation = Validation.CaloTowers.CaloTowersParam_cfi.calotowersAnalyzer.clone()
import Validation.CaloTowers.CaloTowersParam_cfi
HECaloTowersValidation = Validation.CaloTowers.CaloTowersParam_cfi.calotowersAnalyzer.clone()
import Validation.CaloTowers.CaloTowersParam_cfi
HFCaloTowersValidation = Validation.CaloTowers.CaloTowersParam_cfi.calotowersAnalyzer.clone()
import Validation.CaloTowers.CaloTowersParam_cfi
AllCaloTowersValidation = Validation.CaloTowers.CaloTowersParam_cfi.calotowersAnalyzer.clone()
calotowersValidationSequence = cms.Sequence(HBCaloTowersValidation+HECaloTowersValidation+HFCaloTowersValidation+AllCaloTowersValidation)
HBCaloTowersValidation.hcalselector = 'HB'
HBCaloTowersValidation.outputFile = 'CaloTowersValidationHB.root'
HECaloTowersValidation.hcalselector = 'HE'
HECaloTowersValidation.outputFile = 'CaloTowersValidationHE.root'
HFCaloTowersValidation.hcalselector = 'HF'
HFCaloTowersValidation.outputFile = 'CaloTowersValidationHF.root'
AllCaloTowersValidation.hcalselector = 'All'
AllCaloTowersValidation.outputFile = 'CaloTowersValidationAll.root'


