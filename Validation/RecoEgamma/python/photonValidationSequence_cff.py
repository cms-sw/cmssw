import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.tpSelection_cfi import *
from Validation.RecoEgamma.photonValidator_cfi import *
from Validation.RecoEgamma.tkConvValidator_cfi import *


photonValidation.minPhoEtCut = 10
photonValidation.eMax  = 500
photonValidation.etMax = 250
## same for all
photonValidation.convTrackMinPtCut = 1.
photonValidation.useTP = True
photonValidation.rBin = 48
photonValidation.eoverpMin = 0.
photonValidation.eoverpMax = 5.
#

# selectors go in separate "pre-" sequence
photonPrevalidationSequence = cms.Sequence(tpSelection*tpSelecForFakeRate*tpSelecForEfficiency)
photonValidationSequence = cms.Sequence(photonValidation*tkConversionValidation)

