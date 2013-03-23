# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
#from TauAnalysis.MCEmbeddingTools.rerunParticleFlow import rerunParticleFlow, updateInputTags
from TauAnalysis.MCEmbeddingTools.rerunParticleFlow import updateInputTags

def replaceModules(process, moduleOld, modulesNew):
  for sequenceName in process.sequences:
    sequence = getattr(process, sequenceName)
    moduleNameOld = moduleOld.label()
    if moduleNameOld in sequence.moduleNames():
      #print("replacing Module '%s' in Sequence '%s'." % (moduleNameOld, sequenceName))
      sequence.replace(moduleOld, modulesNew)
  for pathName in process.paths:
    path = getattr(process, pathName)
    if moduleNameOld in path.moduleNames():
      #print("replacing Module '%s' in Path '%s'." % (moduleNameOld, pathName))
      path.replace(moduleOld, modulesNew)

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
    process.muonCaloEnergyDepositsByDistance = cms.EDProducer('MuonCaloCleanerByDistance',
      muons = cms.InputTag("muonCaloDistances", "muons"),
      distanceMapMuPlus = cms.InputTag("muonCaloDistances", "distancesMuPlus"),
      distanceMapMuMinus = cms.InputTag("muonCaloDistances", "distancesMuMinus"),
      energyDepositCorrection = cms.PSet(
        H_Ecal_EcalBarrel  = cms.double(process.customization_options.muonCaloCleaningSF.value()*0.9),
        H_Ecal_EcalEndcap  = cms.double(process.customization_options.muonCaloCleaningSF.value()*0.9),   # AB: use barrel value for now
        H_Hcal_HcalBarrel  = cms.double(process.customization_options.muonCaloCleaningSF.value()*1.1), 
        H_Hcal_HcalOuter   = cms.double(process.customization_options.muonCaloCleaningSF.value()*0.8),
        H_Hcal_HcalEndcap  = cms.double(process.customization_options.muonCaloCleaningSF.value()*0.9), 
        H_Hcal_HcalForward = cms.double(process.customization_options.muonCaloCleaningSF.value()*0.000), # CV: simulated tau decay products are not expected to deposit eny energy in HF calorimeter
        H_Hcal_HcalOther   = cms.double(process.customization_options.muonCaloCleaningSF.value()*0.000)
      ),
      verbosity = cms.int32(0)                                                              
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
      verbosity = cms.int32(0)                                                
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
  if not process.customization_options.skipCaloRecHitMixing.value():
    process.castorrecoORG = process.castorreco.clone()
    process.castorreco = cms.EDProducer("CastorRecHitMixer",
      recHitCaloCleanerAllCrossedConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("castorrecoORG"),
          collection2 = cms.InputTag("castorreco", "", inputProcess),
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        )
      ),
      verbosity = cms.int32(0)
    )
    replaceModules(process, process.castorreco, process.castorrecoORG*process.castorreco)
  else:
    print("WARNING: disabling mixing of CASTOR recHit collection, this setting should be used for DEBUGGING only !!")

  # mix recHits in HF calorimeter
  #
  # NOTE: HF calorimeter is not expected to contain any energy from the simulated tau decay products;
  #       the mixing is necessary to get the energy deposits from the Z -> mu+ mu- event
  #       into the embedded event
  #
  if not process.customization_options.skipCaloRecHitMixing.value():
    process.hfrecoORG = process.hfreco.clone()
    process.hfreco = cms.EDProducer("HFRecHitMixer",
      recHitCaloCleanerAllCrossedConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("hfrecoORG"),                              
          collection2 = cms.InputTag("hfreco", "", inputProcess),
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        )
      ),
      verbosity = cms.int32(0)
    )
    replaceModules(process, process.hfreco, process.hfrecoORG*process.hfreco)
  else:
    print("WARNING: disabling mixing of HF recHit collection, this setting should be used for DEBUGGING only !!")

  # mix recHits in preshower
  if not process.customization_options.skipCaloRecHitMixing.value():
    process.ecalPreshowerRecHitORG = process.ecalPreshowerRecHit.clone()
    process.ecalPreshowerRecHit = cms.EDProducer("EcalRecHitMixer",
      recHitCaloCleanerAllCrossedConfig,
      todo = cms.VPSet(
        cms.PSet (
          collection1 = cms.InputTag("ecalPreshowerRecHitORG", "EcalRecHitsES"),                                           
          collection2 = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", inputProcess),
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        )
      ),
      verbosity = cms.int32(0)
    )
    replaceModules(process, process.ecalPreshowerRecHit, process.ecalPreshowerRecHitORG*process.ecalPreshowerRecHit)
  else:
    print("WARNING: disabling mixing of ES recHit collection, this setting should be used for DEBUGGING only !!")

  # mix recHits in ECAL
  if not process.customization_options.skipCaloRecHitMixing.value():
    process.ecalRecHitORG = process.ecalRecHit.clone()
    process.ecalRecHit = cms.EDProducer("EcalRecHitMixer",
      recHitCaloCleanerByDistanceConfig,                                    
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("ecalRecHitORG", "EcalRecHitsEB"),                                   
          collection2 = cms.InputTag("ecalRecHit", "EcalRecHitsEB", inputProcess),
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        ),
        cms.PSet (
          collection1 = cms.InputTag("ecalRecHitORG", "EcalRecHitsEE"),                                  
          collection2 = cms.InputTag("ecalRecHit", "EcalRecHitsEE", inputProcess), 
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        )
      ),
      verbosity = cms.int32(0)
    )
    replaceModules(process, process.ecalRecHit, process.ecalRecHitORG*process.ecalRecHit)
  else:
    print("WARNING: disabling mixing of EB and EE recHit collections, this setting should be used for DEBUGGING only !!")

  # mix recHits in HCAL
  if not process.customization_options.skipCaloRecHitMixing.value():
    process.hbherecoORG = process.hbhereco.clone()
    process.hbhereco = cms.EDProducer("HBHERecHitMixer",
      recHitCaloCleanerByDistanceConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("hbherecoORG", ""),                                
          collection2 = cms.InputTag("hbhereco", "", inputProcess),
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        )
      ),
      verbosity = cms.int32(0)
    )
    replaceModules(process, process.hbhereco, process.hbherecoORG*process.hbhereco)
    
    process.horecoORG = process.horeco.clone()
    process.horeco = cms.EDProducer("HORecHitMixer",
      recHitCaloCleanerByDistanceConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("horecoORG", ""),                              
          collection2 = cms.InputTag("horeco", "", inputProcess),
          killNegEnergyBeforeMixing1 = cms.bool(False),
          killNegEnergyBeforeMixing2 = cms.bool(True),                                  
          muonEnSutractionMode = cms.string("subtractFromCollection2BeforeMixing"),
          killNegEnergyAfterMixing = cms.bool(False)
        )
      ),
      verbosity = cms.int32(0)
    )
    replaceModules(process, process.horeco, process.horecoORG*process.horeco)
  else:
    print("WARNING: disabling mixing of HB, HE and HO recHit collections, this setting should be used for DEBUGGING only !!")

  # CV: Compute hits in muon detectors of the two muons produced in Z -> mu+ mu- decay
  process.muonDetHits = cms.EDProducer('MuonDetCleaner',
    trackAssociator = trackAssocParamsForMuonCleaning,
    selectedMuons = process.customization_options.ZmumuCollection,                                       
    verbosity = cms.int32(0)
  )
  if process.customization_options.replaceGenOrRecMuonMomenta.value() == "gen":
    process.muonDetHits.trackAssociator.muonMaxDistanceX = cms.double(1.e+3)
    process.muonDetHits.trackAssociator.muonMaxDistanceX = cms.double(1.e+3)
    process.muonDetHits.trackAssociator.dRMuonPreselection = cms.double(0.5)    
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
  replaceModules(process, process.csc2DRecHits, process.csc2DRecHitsORG*process.csc2DRecHits)
  
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
  replaceModules(process, process.dt1DRecHits, process.dt1DRecHitsORG*process.dt1DRecHits)
  
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
  replaceModules(process, process.rpcRecHits, process.rpcRecHitsORG*process.rpcRecHits)
    
  return process
