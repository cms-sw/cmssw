import FWCore.ParameterSet.Config as cms

from Validation.MtdValidation.MtdTracksPostProcessor_cfi import MtdTracksPostProcessor as _MtdTracksPostProcessor
from Validation.MtdValidation.Primary4DVertexPostProcessor_cfi import Primary4DVertexPostProcessor as _Primary4DVertexPostProcessor

hltMtdTracksPostProcessor = _MtdTracksPostProcessor.clone(
    folder = cms.string('HLT/MTD/Tracks/')
)
hltPrimary4DVertexPostProcessor = _Primary4DVertexPostProcessor.clone(
    folder = cms.string('HLT/MTD/Vertices/')
)

hltMtdValidationPostProcessor = cms.Sequence(#hltMtdTracksPostProcessor +
                                             hltPrimary4DVertexPostProcessor)
