import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.v0validator_cfi import *

vertexValidation = cms.Sequence(v0Validator)
