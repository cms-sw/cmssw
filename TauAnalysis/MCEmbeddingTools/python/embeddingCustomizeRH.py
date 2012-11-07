# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
#from TauAnalysis.MCEmbeddingTools.rerunParticleFlow import rerunParticleFlow, updateInputTags
from TauAnalysis.MCEmbeddingTools.rerunParticleFlow import updateInputTags

def customise(process, inputProcess):

  # Determine detIds of calorimeter cells crossed by muon track
  trackAssocParamsForMuonCleaning = TrackAssociatorParameterBlock.TrackAssociatorParameters
  updateInputTags(process, trackAssocParamsForMuonCleaning, inputProcess)
  process.muonCaloEnergyDepositsAllCrossed = cms.EDProducer('MuonCaloCleanerAllCrossed',
    trackAssociator = trackAssocParamsForMuonCleaning,
    selectedMuons = process.customization_options.ZmumuCollection,
    esRecHits = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", inputProcess)
  )
  process.ProductionFilterSequence += process.muonCaloEnergyDepositsAllCrossed

  recHitCaloCleanerAllCrossedConfig = cms.PSet(
    srcEnergyDepositMapMuPlus = cms.InputTag("muonCaloEnergyDepositsAllCrossed", "energyDepositsMuPlus"),
    srcEnergyDepositMapMuMinus = cms.InputTag("muonCaloEnergyDepositsAllCrossed", "energyDepositsMuMinus"),
    typeEnergyDepositMap = cms.string("absolute"), # CV: use 'absolute' for 'MuonCaloCleanerAllCrossed' module
  )

  recHitCaloCleanerByDistanceConfig = None
  if process.customization_options.cleaningMode == 'DEDX':
    # CV: Compute expected energy deposits of muon in EB/EE, HB/HE and HO:
    #      (1) compute distance traversed by muons produced in Z -> mu+ mu- decay
    #          through individual calorimeter cells
    #      (2) scale distances by expected energy loss dE/dx of muon
    process.muonCaloDistances = cms.EDProducer('MuonCaloDistanceProducer',
      trackAssociator = TrackAssociatorParameterBlock.TrackAssociatorParameters,
      selectedMuons = process.customization_options.ZmumuCollection
    )
    process.ProductionFilterSequence += process.muonCaloDistances

    process.muonCaloEnergyDepositsByDistance = cms.EDProducer('MuonCaloCleanerByDistance',
      muons = cms.InputTag("muonCaloDistances", "muons"),
      distanceMapMuPlus = cms.InputTag("muonCaloDistances", "distancesMuPlus"),
      distanceMapMuMinus = cms.InputTag("muonCaloDistances", "distancesMuMinus"),
      energyDepositCorrection = cms.PSet(
        H_Ecal_EcalBarrel  = cms.double(0.9),
        H_Ecal_EcalEndcap  = cms.double(0.9), # use barrel value for now
        H_Hcal_HcalBarrel  = cms.double(1.1), 
        H_Hcal_HcalOuter   = cms.double(0.8),
        H_Hcal_HcalEndcap  = cms.double(0.9), 
        H_Hcal_HcalForward = cms.double(0.000), # CV: simulated tau decay products are not expected to deposit eny energy in HF calorimeter
        H_Hcal_HcalOther   = cms.double(0.000)
      )                                                  
    )
    process.ProductionFilterSequence += process.muonCaloEnergyDepositsByDistance
  
    recHitCaloCleanerByDistanceConfig = cms.PSet(
      srcEnergyDepositMapMuPlus = cms.InputTag("muonCaloEnergyDepositsByDistance", "energyDepositsMuPlus"),
      srcEnergyDepositMapMuMinus = cms.InputTag("muonCaloEnergyDepositsByDistance", "energyDepositsMuMinus"),
      typeEnergyDepositMap = cms.string("absolute"), # CV: use 'absolute' ('relative') for 'MuonCaloCleanerByDistance' ('PFMuonCleaner') module
    )
  elif process.customization_options.cleaningMode == 'PF':
    # Take energy deposits associated to muon by particle-flow algorithm
    process.pfMuonCaloEnergyDeposits = cms.EDProducer('PFMuonCaloCleaner',
      selectedMuons = process.customization_options.ZmumuCollection,
      pfCandidates = cms.InputTag("particleFlowForPFMuonCleaning"),
      dRmatch = cms.double(0.3),
      verbosity = cms.int32(1)                                                
    )
    process.ProductionFilterSequence += process.pfMuonCaloEnergyDeposits
    
    recHitCaloCleanerByDistanceConfig = cms.PSet(
      srcEnergyDepositMapMuPlus = cms.InputTag("pfMuonCaloEnergyDeposits", "energyDepositsMuPlus"),
      srcEnergyDepositMapMuMinus = cms.InputTag("pfMuonCaloEnergyDeposits", "energyDepositsMuMinus"),
      typeEnergyDepositMap = cms.string("absolute"), # CV: use 'absolute' for 'PFMuonCaloCleaner'
    )
  else:
    raise ValueError("Invalid Configuration parameter 'cleaningMode' = %s !!" % process.customization_options.cleaningMode)

  # mix recHits in CASTOR calorimeter
  #
  # NOTE: CASTOR is not expected to contain any energy from the simulated tau decay products;
  #       the mixing is necessary to get the energy deposits from the Z -> mu+ mu- event
  #       into the embedded event
  #
  process.castorrecoORG = process.castorreco.clone()
  process.castorreco = cms.EDProducer("CastorRecHitMixer",
    recHitCaloCleanerAllCrossedConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("castorreco", "", inputProcess),
        collection2 = cms.InputTag("castorrecoORG")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "castorreco" in pth.moduleNames():
      pth.replace(process.castorreco, process.castorrecoORG*process.castorreco)

  # mix recHits in preshower 
  process.ecalPreshowerRecHitORG = process.ecalPreshowerRecHit.clone()
  process.ecalPreshowerRecHit = cms.EDProducer("EcalRecHitMixer",
    recHitCaloCleanerAllCrossedConfig,
    todo = cms.VPSet(
      cms.PSet (
        collection1 = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", inputProcess),
        collection2 = cms.InputTag("ecalPreshowerRecHitORG", "EcalRecHitsES")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "ecalPreshowerRecHit" in pth.moduleNames():
      pth.replace(process.ecalPreshowerRecHit, process.ecalPreshowerRecHitORG*process.ecalPreshowerRecHit)

  # mix recHits in ECAL
  process.ecalRecHitORG = process.ecalRecHit.clone()
  process.ecalRecHit = cms.EDProducer("EcalRecHitMixer",
    recHitCaloCleanerByDistanceConfig,                                    
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("ecalRecHit", "EcalRecHitsEB", inputProcess), 
        collection2 = cms.InputTag("ecalRecHitORG", "EcalRecHitsEB")
      ),
      cms.PSet (
        collection1 = cms.InputTag("ecalRecHit", "EcalRecHitsEE", inputProcess), 
        collection2 = cms.InputTag("ecalRecHitORG", "EcalRecHitsEE")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "ecalRecHit" in pth.moduleNames():
      pth.replace(process.ecalRecHit, process.ecalRecHitORG*process.ecalRecHit)

  # mix recHits in HCAL
  process.hbherecoORG = process.hbhereco.clone()
  process.hbhereco = cms.EDProducer("HBHERecHitMixer",
    recHitCaloCleanerByDistanceConfig,
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
  process.horeco = cms.EDProducer("HORecHitMixer",
    recHitCaloCleanerByDistanceConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("horeco", "", inputProcess),
        collection2 = cms.InputTag("horecoORG", "")
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "horeco" in pth.moduleNames():
      pth.replace(process.horeco, process.horecoORG*process.horeco)

  # CV: Compute hits in muon detectors of the two muons produced in Z -> mu+ mu- decay
  process.muonDetHits = cms.EDProducer('MuonDetCleaner',
    selectedMuons = process.customization_options.ZmumuCollection,
    verbosity = cms.int32(0)
  )
  process.ProductionFilterSequence += process.muonDetHits

  recHitMuonDetCleanerConfig = cms.PSet(
    srcHitMapMuPlus = cms.InputTag("muonDetHits", "hitsMuPlus"),
    srcHitMapMuMinus = cms.InputTag("muonDetHits", "hitsMuMinus"),
    verbosity = cms.int32(0)
  )  

  # mix recHits in CSC
  process.csc2DRecHitsORG = process.csc2DRecHits.clone()
  process.csc2DRecHits = cms.EDProducer("CSCRecHitMixer",
    recHitMuonDetCleanerConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("csc2DRecHits", "", inputProcess),
        cleanCollection1 = cms.bool(True),                                    
        collection2 = cms.InputTag("csc2DRecHitsORG", ""),
        cleanCollection2 = cms.bool(False)                                    
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "csc2DRecHits" in pth.moduleNames():
      pth.replace(process.csc2DRecHits, process.csc2DRecHitsORG*process.csc2DRecHits)

  # mix recHits in DT
  process.dt1DRecHitsORG = process.dt1DRecHits.clone()
  process.dt1DRecHits = cms.EDProducer("DTRecHitMixer",
    recHitMuonDetCleanerConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("dt1DRecHits", "", inputProcess),
        cleanCollection1 = cms.bool(True),                                    
        collection2 = cms.InputTag("dt1DRecHitsORG", ""),
        cleanCollection2 = cms.bool(False)                                     
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "dt1DRecHits" in pth.moduleNames():
      pth.replace(process.dt1DRecHits, process.dt1DRecHitsORG*process.dt1DRecHits)

  # mix recHits in RPC
  process.rpcRecHitsORG = process.rpcRecHits.clone()
  process.rpcRecHits = cms.EDProducer("RPCRecHitMixer",
    recHitMuonDetCleanerConfig,
    todo = cms.VPSet(
      cms.PSet(
        collection1 = cms.InputTag("rpcRecHits", "", inputProcess),
        cleanCollection1 = cms.bool(True),                                    
        collection2 = cms.InputTag("rpcRecHitsORG", ""),
        cleanCollection2 = cms.bool(False)                                    
      )
    )
  )
  for p in process.paths:
    pth = getattr(process, p)
    if "rpcRecHits" in pth.moduleNames():
      pth.replace(process.rpcRecHits, process.rpcRecHitsORG*process.rpcRecHits)
  
  return process
