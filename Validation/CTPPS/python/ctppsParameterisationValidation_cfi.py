import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.paramValidation_cfi import paramValidation

ctppsParameterisationValidation = cms.Sequence( paramValidation )
