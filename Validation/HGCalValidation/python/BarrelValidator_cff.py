import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
from Validation.HGCalValidation.barrelValidator_cfi import barrelValidator as _barrelValidator

barrelValidator = _barrelValidator.clone()
