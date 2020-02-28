import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.tpSelection_cfi import *
from Validation.RecoEgamma.photonValidator_cfi import *
from Validation.RecoEgamma.tkConvValidator_cfi import *

import Validation.RecoEgamma.photonValidator_cfi

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
pfPhotonValidation = Validation.RecoEgamma.photonValidator_cfi.photonValidation.clone()
pfPhotonValidation.ComponentName = cms.string('pfPhotonValidation')
pfPhotonValidation.OutputFileName = cms.string('PFPhotonValidationHistos.root')
pfPhotonValidation.phoProducer = cms.string('gedPhotons')
pfPhotonValidation.photonCollection = cms.string('')
pfPhotonValidation.analyzerName = cms.string('pfPhotonValidator')
pfPhotonValidation.minPhoEtCut = 10
pfPhotonValidation.eMax  = 500
pfPhotonValidation.etMax = 250
## same for all
pfPhotonValidation.convTrackMinPtCut = 1.
pfPhotonValidation.useTP = True
pfPhotonValidation.rBin = 48
pfPhotonValidation.eoverpMin = 0.
pfPhotonValidation.eoverpMax = 5.
#
oldpfPhotonValidation = Validation.RecoEgamma.photonValidator_cfi.photonValidation.clone()
oldpfPhotonValidation.ComponentName = cms.string('oldpfPhotonValidation')
oldpfPhotonValidation.OutputFileName = cms.string('oldPFPhotonValidationHistos.root')
oldpfPhotonValidation.phoProducer = cms.string('pfPhotonTranslator')
oldpfPhotonValidation.photonCollection = cms.string('pfphot')
oldpfPhotonValidation.analyzerName = cms.string('oldpfPhotonValidator')
oldpfPhotonValidation.minPhoEtCut = 10
oldpfPhotonValidation.eMax  = 500
oldpfPhotonValidation.etMax = 250
## same for all
oldpfPhotonValidation.convTrackMinPtCut = 1.
oldpfPhotonValidation.useTP = True
oldpfPhotonValidation.rBin = 48
oldpfPhotonValidation.eoverpMin = 0.
oldpfPhotonValidation.eoverpMax = 5.

import Validation.RecoEgamma.tkConvValidator_cfi



# selectors go in separate "pre-" sequence
photonPrevalidationSequence = cms.Sequence(tpSelection*tpSelecForFakeRate*tpSelecForEfficiency)
photonValidationSequence = cms.Sequence(trackAssociatorByHitsForPhotonValidation*photonValidation*pfPhotonValidation*trackAssociatorByHitsForConversionValidation*tkConversionValidation)


from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( photonValidation, useTP = cms.bool(False) )
phase2_common.toModify( pfPhotonValidation, useTP = cms.bool(False) )
phase2_common.toModify( oldpfPhotonValidation, useTP = cms.bool(False) )
phase2_common.toModify( tkConversionValidation, useTP = cms.bool(False) )


