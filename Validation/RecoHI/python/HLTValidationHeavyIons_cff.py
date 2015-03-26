import FWCore.ParameterSet.Config as cms

from Validation.RecoHI.SingleJetValidationHI_cfi import *
from Validation.RecoHI.EgammaValidationHI_cff import *

hltPrevalidationHI = cms.Sequence( hiEgammaPrevalidationSequence )

hltValidationHI = cms.Sequence(hiSingleJetValidation
                               + hiEgammaValidationSequence
                               )

hltValidationHI.remove(hiSingleJetValidation)
