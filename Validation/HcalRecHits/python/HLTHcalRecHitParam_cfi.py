import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *

hltHCALRecoAnalyzer = hcalRecoAnalyzer.clone()
hltHCALRecoAnalyzer.TopFolderName             = cms.string('HLT/RecHits/Simulation')
hltHCALRecoAnalyzer.outputFile                = cms.untracked.string('')
hltHCALRecoAnalyzer.HBHERecHitCollectionLabel = cms.untracked.InputTag("hltHbhereco")
hltHCALRecoAnalyzer.HFRecHitCollectionLabel   = cms.untracked.InputTag("hltHfreco")
hltHCALRecoAnalyzer.HORecHitCollectionLabel   = cms.untracked.InputTag("hltHoreco")
hltHCALRecoAnalyzer.EBRecHitCollectionLabel   = cms.InputTag("hltEcalRecHit","EcalRecHitsEB")
hltHCALRecoAnalyzer.EERecHitCollectionLabel   = cms.InputTag("hltEcalRecHit","EcalRecHitsEE")
hltHCALRecoAnalyzer.ecalselector              = cms.untracked.string('yes')
hltHCALRecoAnalyzer.hcalselector              = cms.untracked.string('all')
hltHCALRecoAnalyzer.mc                        = cms.untracked.string('yes')
hltHCALRecoAnalyzer.SimHitCollectionLabel = cms.untracked.InputTag("g4SimHits","HcalHits")
hltHCALRecoAnalyzer.TestNumber                = cms.bool(False)


hltHCALNoiseRates = hcalNoiseRates.clone()
hltHCALNoiseRates.outputFile   = cms.untracked.string('')
hltHCALNoiseRates.rbxCollName  = cms.untracked.InputTag('hcalnoise')
hltHCALNoiseRates.minRBXEnergy = cms.untracked.double(20.0)
hltHCALNoiseRates.minHitEnergy = cms.untracked.double(1.5)
hltHCALNoiseRates.noiselabel   = cms.InputTag('hcalnoise')


from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hltHCALRecoAnalyzer, TestNumber = cms.bool(True) )

#from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
#phase2_hcal.toModify(hcalRecoAnalyzer, mc = cms.untracked.string('no') )
