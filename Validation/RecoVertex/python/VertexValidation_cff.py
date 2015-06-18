import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.v0validator_cfi import *
from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *

# Rely on tracksValidationTruth sequence being already run
vertexValidation = cms.Sequence(v0Validator
                                * vertexAnalysisSequence)


from Validation.RecoTrack.TrackValidation_cff import tracksValidationTruth
vertexValidationStandalone = cms.Sequence(
    tracksValidationTruth
    * vertexValidation
)
