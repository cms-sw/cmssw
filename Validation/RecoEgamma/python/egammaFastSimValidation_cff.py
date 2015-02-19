import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.electronValidationSequence_cff import *
from Validation.RecoEgamma.photonValidationSequence_cff import *

photonValidation.isRunCentrally = True
photonValidation.fastSim = True

pfPhotonValidation.isRunCentrally = True
pfPhotonValidation.fastSim = True

egammaFastSimValidation = cms.Sequence(trackAssociatorByHitsForPhotonValidation*photonValidation*pfPhotonValidation)
