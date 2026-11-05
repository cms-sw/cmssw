import FWCore.ParameterSet.Config as cms
from Validation.RecoVertex.PostProcessorSecondaryVertex_cff import *

hltSecondaryVertexPostProcessor = postProcessorSecondaryVertex.clone(
    subDirs = ("HLT/SecondaryVertices/Validation/*",)
)

hltSecondaryVertexPostProcessorPerPdg = postProcessorSecondaryVertexPerPdg.clone(
    subDirs = ("HLT/SecondaryVertices/Validation/*",)
)

hltSecondaryVertexPostProcessorSummary = postProcessorSecondaryVertexSummary.clone(
    subDirs = ("HLT/SecondaryVertices/Validation/*",)
)

HLTSecondaryVertexPostProcessorSequence = cms.Sequence(
    hltSecondaryVertexPostProcessor +
    hltSecondaryVertexPostProcessorPerPdg +
    hltSecondaryVertexPostProcessorSummary
)
