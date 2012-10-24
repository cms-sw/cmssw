# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

def customise(process, inputProcess):

  # CV: Compute expected energy deposits of muon in EB/EE, HB/HE and HO:
  #      (1) compute distance traversed by muons produced in Z -> mu+ mu- decay
  #          through individual calorimeter cells
  #      (2) scale distances by expected energy loss dE/dx of muon
  process.muonCaloDistances = cms.EDProducer('MuonCaloDistanceProducer',
    trackAssociator = TrackAssociatorParameterBlock.TrackAssociatorParameters,
    selectedMuons = process.customization_options.ZmumuCollection
  )
  process.ProductionFilterSequence += process.muonCaloDistances

  process.muonCaloEnergyDeposits = cms.EDProducer('MuonCaloCleanerByDistance',
    distanceMapMuPlus = cms.InputTag("muonCaloDistances", "distanceMuPlus"),
    distanceMapMuMinus = cms.InputTag("muonCaloDistances", "distanceMuMinus"),
    energyDepositPerDistance = cms.PSet(
      H_Ecal_EcalBarrel  = cms.double(0.020), # GeV per cm (assuming density of all calorimeters to be ~10g/cm^3)
      H_Ecal_EcalEndcap  = cms.double(0.020),   
      H_Hcal_HcalBarrel  = cms.double(0.020),   
      H_Hcal_HcalOuter   = cms.double(0.020),
      H_Hcal_HcalEndcap  = cms.double(0.020),
      H_Hcal_HcalForward = cms.double(0.000), # CV: simulated tau decay products are not expected to deposit eny energy in HF calorimeter
      H_Hcal_HcalOther   = cms.double(0.000)
    )                                                  
  )
  process.ProductionFilterSequence += process.muonCaloEnergyDeposits
  
  recHitCleanerConfig = cms.PSet(
    srcEnergyDepositMapMuPlus = cms.InputTag("muonCaloEnergyDeposits", "energyDepositMuPlus"),
    srcEnergyDepositMapMuMinus = cms.InputTag("muonCaloEnergyDeposits", "energyDepositMuMinus"),
    typeEnergyDepositMap = cms.string("absolute"), # CV: use 'absolute' ('relative') for 'MuonCaloCleanerByDistance' ('PFMuonCleaner') modules
  )

  # mix recHits in CASTOR calorimeter
  #
  # NOTE: CASTOR is not expected to contain any energy from the simulated tau decay products;
  #       the mixing is necessary to get the energy deposits from the Z -> mu+ mu- event
  #       into the embedded event
  #
  process.castorrecoORG = process.castorreco.clone()
  process.castorreco = cms.EDProducer("CastorRHMixer",
    recHitCleanerConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("castorreco", "", inputProcess),
        collection2 = cms.InputTag("castorrecoORG")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "castorreco" in pth.moduleNames():
      pth.replace(process.castorreco, process.castorrecoORG*process.castorreco)

  # mix recHits in preshower 
  ##process.ecalPreshowerRecHitORG = process.ecalPreshowerRecHit.clone()
  ##process.ecalPreshowerRecHit = cms.EDProducer("EcalRHMixer",
  ##  recHitCleanerConfig,
  ##  todo = cms.VPSet(
  ##    cms.PSet (
  ##      collection1 = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", inputProcess),
  ##      collection2 = cms.InputTag("ecalPreshowerRecHitORG", "EcalRecHitsES")
  ##    )
  ##  )
  ##)
  ##for p in process.paths:
  ##  pth = getattr(process,p)
  ##  if "ecalPreshowerRecHit" in pth.moduleNames():
  ##    pth.replace(process.ecalPreshowerRecHit, process.ecalPreshowerRecHitORG*process.ecalPreshowerRecHit)

  # mix recHits in ECAL
  process.ecalRecHitORG = process.ecalRecHit.clone()
  process.ecalRecHit = cms.EDProducer("EcalRHMixer",
    recHitCleanerConfig,                                    
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("ecalRecHit","EcalRecHitsEB", inputProcess ), 
        collection2 = cms.InputTag("ecalRecHitORG","EcalRecHitsEB" )
      ),
      cms.PSet (
        collection1 = cms.InputTag("ecalRecHit","EcalRecHitsEE", inputProcess), 
        collection2 = cms.InputTag("ecalRecHitORG","EcalRecHitsEE")
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
    recHitCleanerConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("hbhereco", "", inputProcess),
        collection2 = cms.InputTag("hbherecoORG", "")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "hbhereco" in pth.moduleNames():
      pth.replace(process.hbhereco, process.hbherecoORG*process.hbhereco)

  process.horecoORG = process.horeco.clone()
  process.horeco = cms.EDProducer("HORHMixer",
    recHitCleanerConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("horeco", "", inputProcess),
        collection2 = cms.InputTag("horecoORG", "")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process,p)
    if "horeco" in pth.moduleNames():
      pth.replace(process.horeco, process.horecoORG*process.horeco)

  return process
