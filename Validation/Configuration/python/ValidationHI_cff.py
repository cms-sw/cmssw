import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Validation_cff import *
from Validation.RecoHI.hiBasicGenTest_cfi import *
from Validation.RecoHI.globalValidationHeavyIons_cff import *
from Validation.RecoHI.HLTValidationHeavyIons_cff import *
from Validation.EventGenerator.BasicHepMCHeavyIonValidation_cfi import *

validationHI = cms.Sequence(hiBasicGenTest
                            *basicHepMCHeavyIonValidation
                            *globaldigisanalyze
                            *globalhitsanalyze
                            *globalrechitsanalyze
                            *globalValidationHI
                            *hltValidationHI
                            )

# temporary removal
# due to massive redundant output
validationHI.remove(condDataValidation)
