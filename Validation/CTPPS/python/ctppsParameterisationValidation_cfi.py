import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.paramValidation_cfi import paramValidation
from Validation.CTPPS.scoringPlaneValidation_cfi import scoringPlaneValidation

ctppsParameterisationValidation = cms.Sequence(
    paramValidation
    * scoringPlaneValidation
)
