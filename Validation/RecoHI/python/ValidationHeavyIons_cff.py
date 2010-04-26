import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Validation_cff import *
from Validation.RecoHI.hiBasicGenTest_cfi import *
from Validation.RecoHI.globalValidationHeavyIons_cff import *
from Validation.RecoHI.HLTValidationHeavyIons_cff import *

validationHeavyIons = cms.Sequence(cms.SequencePlaceholder("mix")
                                   +hiBasicGenTest
                                   *globaldigisanalyze
                                   *globalhitsanalyze
                                   *globalrechitsanalyze
                                   *globalValidationHI
                                   *hltValidationHI
                                   )
