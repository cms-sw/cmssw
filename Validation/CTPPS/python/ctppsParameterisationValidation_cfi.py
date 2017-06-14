import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.ctppsParameterisation_cfi import ctppsParameterisation

ctppsParameterisationValidation = cms.Sequence( ctppsParameterisation )
