import FWCore.ParameterSet.Config as cms

from Validation.TrackerHits.trackerHitsValidation_cff import *
from Validation.TrackerDigis.trackerDigisValidation_cff import *
from Validation.TrackerRecHits.trackerRecHitsValidation_cff import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.SiTrackingRecHitsValid_cff import *
from Validation.RecoTrack.TrackValidation_cff import *
multiTrackValidator.UseAssociators = True
from Validation.EcalHits.ecalSimHitsValidationSequence_cff import *
from Validation.EcalDigis.ecalDigisValidationSequence_cff import *
from Validation.EcalRecHits.ecalRecHitsValidationSequence_cff import *
from Validation.HcalHits.HcalSimHitStudy_cfi import *
#from Validation.HcalDigis.hcalDigisValidationSequence_cff import *
from Validation.HcalRecHits.hcalRecHitsValidationSequence_cff import *
from Validation.CaloTowers.calotowersValidationSequence_cff import *
from Validation.MuonHits.muonHitsValidation_cfi import *
from Validation.MuonDTDigis.dtDigiValidation_cfi import *
from Validation.MuonCSCDigis.cscDigiValidation_cfi import *
from Validation.MuonRPCDigis.validationMuonRPCDigis_cfi import *
from Validation.RecoMuon.muonValidation_cff import *
#from Validation.MuonIsolation.MuIsoVal_cff import *

globalValidation = cms.Sequence(trackerHitsValidation+trackerDigisValidation+trackerRecHitsValidation+trackingTruthValid+trackingRecHitsValid+tracksValidation+
                                ecalSimHitsValidationSequence+ecalDigisValidationSequence+ecalRecHitsValidationSequence+
                                hcalSimHitStudy+hcalRecHitsValidationSequence+calotowersValidationSequence+
                                validSimHit+muondtdigianalyzer+cscDigiValidation+validationMuonRPCDigis+recoMuonValidation)#+muIsoVal_seq)

globalValidation_pu = cms.Sequence(trackerHitsValidation+trackerDigisValidation+trackingTruthValid+tracksValidation+
                                   ecalSimHitsValidationSequence+hcalSimHitStudy+hcalRecHitsValidationSequence+calotowersValidationSequence+
                                   validSimHit+muondtdigianalyzer+validationMuonRPCDigis+recoMuonValidation)

