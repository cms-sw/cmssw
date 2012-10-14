# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

def customise(process, inputProcess):

  # note - we should probably check this
  recHitCleanerConfig_crossed = cms.PSet(
    depsPlus = cms.InputTag("anaDeposits", "plus" ),
    depsMinus = cms.InputTag("anaDeposits", "minus" ),
    ZmumuCands = process.customization_options.ZmumuCollection
  ) 

  recHitCleanerConfig_const = cms.PSet(
    depsPlus = recHitCleanerConfig_crossed.depsPlus,
    depsMinus = recHitCleanerConfig_crossed.depsMinus,
    ZmumuCands = recHitCleanerConfig_crossed.ZmumuCands,
    names = cms.vstring(
      "H_Ecal_EcalBarrel",
      "H_Ecal_EcalEndcap",
      "H_Hcal_HcalBarrel",
      "H_Hcal_HcalOuter",
      "H_Hcal_HcalEndcap"
    ),
    H_Ecal_EcalBarrel = cms.double(0.2),
    H_Ecal_EcalEndcap = cms.double(0.),   
    H_Hcal_HcalBarrel = cms.double(6.),   
    H_Hcal_HcalOuter  = cms.double(6.),
    H_Hcal_HcalEndcap = cms.double(6.)
  )
    
  # mix recHits in castor calorimeter
  process.castorrecoORG = process.castorreco.clone()
  process.castorreco = cms.EDProducer("CastorRHMixer",
    cleaningAlgo = cms.string("CaloCleanerAllCrossed"),
    cleaningConfig = recHitCleanerConfig_crossed,
    todo = cms.VPSet(
      cms.PSet(
        colZmumu = cms.InputTag("castorreco", "", inputProcess),
        colTauTau = cms.InputTag("castorrecoORG")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "castorreco" in pth.moduleNames():
      pth.replace(process.castorreco, process.castorrecoORG*process.castorreco)

  # mix recHits in preshower 
  process.ecalPreshowerRecHitORG = process.ecalPreshowerRecHit.clone()
  process.ecalPreshowerRecHit = cms.EDProducer("EcalRHMixer",
    cleaningAlgo = cms.string("CaloCleanerAllCrossed"), # CaloCleanerMVA
    cleaningConfig = recHitCleanerConfig_crossed,
    todo = cms.VPSet(
      cms.PSet (
        colZmumu = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", inputProcess),
        colTauTau = cms.InputTag("ecalPreshowerRecHitORG", "EcalRecHitsES")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "ecalPreshowerRecHit" in pth.moduleNames():
      pth.replace(process.ecalPreshowerRecHit, process.ecalPreshowerRecHitORG*process.ecalPreshowerRecHit)

  # mix recHits in ECAL
  process.ecalRecHitORG = process.ecalRecHit.clone()
  process.ecalRecHit = cms.EDProducer("EcalRHMixer",
    cleaningAlgo = cms.string("CaloCleanerConst"), 
    cleaningConfig = recHitCleanerConfig_const,
    todo = cms.VPSet(
      cms.PSet(
        colZmumu = cms.InputTag("ecalRecHit","EcalRecHitsEB", inputProcess ), 
        colTauTau = cms.InputTag("ecalRecHitORG","EcalRecHitsEB" )
      ),
      cms.PSet (
        colZmumu = cms.InputTag("ecalRecHit","EcalRecHitsEE", inputProcess), 
        colTauTau = cms.InputTag("ecalRecHitORG","EcalRecHitsEE")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "ecalRecHit" in pth.moduleNames():
      pth.replace(process.ecalRecHit, process.ecalRecHitORG*process.ecalRecHit)

  # mix recHits in HCAL
  process.hbherecoORG = process.hbhereco.clone()
  process.hbhereco = cms.EDProducer("HBHERHMixer",
    cleaningConfig = recHitCleanerConfig_const,
    cleaningAlgo = cms.string("CaloCleanerConst"), 
    todo = cms.VPSet(
      cms.PSet(
        colZmumu = cms.InputTag("hbhereco", "", inputProcess),
        colTauTau = cms.InputTag("hbherecoORG", "")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "hbhereco" in pth.moduleNames():
      pth.replace(process.hbhereco, process.hbherecoORG*process.hbhereco)

  # mix recHits in HF
  process.hfrecoORG = process.hfreco.clone()
  process.hfreco = cms.EDProducer("HFRHMixer",
    cleaningConfig = recHitCleanerConfig_const,
    cleaningAlgo = cms.string("CaloCleanerConst"),       
    todo = cms.VPSet(
      cms.PSet(
        colZmumu = cms.InputTag("hfreco", "", inputProcess),
        colTauTau = cms.InputTag("hfrecoORG", "")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "hfreco" in pth.moduleNames():
      pth.replace(process.hfreco, process.hfrecoORG*process.hfreco)

  process.horecoORG = process.horeco.clone()
  process.horeco = cms.EDProducer("HORHMixer",
    cleaningConfig = recHitCleanerConfig_const,
    cleaningAlgo = cms.string("CaloCleanerConst"), 
    todo = cms.VPSet(
      cms.PSet(
        colZmumu = cms.InputTag("horeco", "", inputProcess),
        colTauTau = cms.InputTag("horecoORG", "")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "horeco" in pth.moduleNames():
      pth.replace(process.horeco, process.horecoORG*process.horeco)

  return process
