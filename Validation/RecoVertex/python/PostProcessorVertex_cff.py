import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.PostProcessorV0_cfi import *
from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_Client_cfi import *


postProcessorVertexSequence = cms.Sequence(
    postProcessorVertex +
    postProcessorV0
)

postProcessorVertexStandAlone = cms.Sequence(postProcessorVertexSequence)
