# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

import PhysicsTools.PatAlgos.tools.helpers as configtools
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
from TauAnalysis.MCEmbeddingTools.rerunParticleFlow import updateInputTags

def replaceModule_or_Sequence(process, moduleOld, modulesNew):
  for sequenceName in process.sequences:    
    sequence = getattr(process, sequenceName)
    #print("sequence = %s: modules = %s" % (sequenceName, sequence.moduleNames())) 
    moduleNameOld = moduleOld.label()
    if moduleNameOld in sequence.moduleNames():
      #print("replacing Module '%s' in Sequence '%s'." % (moduleNameOld, sequenceName))
      sequence.replace(moduleOld, modulesNew)
  for pathName in process.paths:
    path = getattr(process, pathName)
    #print("path = %s: modules = %s" % (pathName, path.moduleNames())) 
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
    replaceModule_or_Sequence(process, process.castorreco, process.castorrecoORG*process.castorreco)
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
    replaceModule_or_Sequence(process, process.hfreco, process.hfrecoORG*process.hfreco)
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
    replaceModule_or_Sequence(process, process.ecalPreshowerRecHit, process.ecalPreshowerRecHitORG*process.ecalPreshowerRecHit)
  else:
    print("WARNING: disabling mixing of ES recHit collection, this setting should be used for DEBUGGING only !!")

  # mix recHits in ECAL
  if not process.customization_options.skipCaloRecHitMixing.value():
    print "Mixing ECAL recHit collections"
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
    replaceModule_or_Sequence(process, process.ecalRecHit, process.ecalRecHitORG*process.ecalRecHit)
  else:
    print("WARNING: disabling mixing of EB and EE recHit collections, this setting should be used for DEBUGGING only !!")

  # mix recHits in HCAL
  if not process.customization_options.skipCaloRecHitMixing.value():
    print "Mixing HCAL recHit collection"
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
    replaceModule_or_Sequence(process, process.hbhereco, process.hbherecoORG*process.hbhereco)
    
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
    replaceModule_or_Sequence(process, process.horeco, process.horecoORG*process.horeco)
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

  if process.customization_options.muonMixingMode.value() == 2:
    # CV: clone muon sequence for muon track segment reconstruction
    configtools.cloneProcessingSnippet(process, process.muonlocalreco, "ORG")
    process.reconstruction_step.replace(process.dt1DRecHits, process.muonlocalrecoORG*process.dt1DRecHits)

  if process.customization_options.muonMixingMode.value() == 1 or \
     process.customization_options.muonMixingMode.value() == 2:
    # mix recHits in CSC
    print "Mixing CSC recHit collection"
    process.csc2DRecHitsORG = process.csc2DRecHits.clone()
    process.csc2DRecHits = cms.EDProducer("CSCRecHitMixer",
      recHitMuonDetCleanerConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("csc2DRecHitsORG", ""),
          cleanCollection1 = cms.bool(False),                                     
          collection2 = cms.InputTag("csc2DRecHits", "", inputProcess),
          cleanCollection2 = cms.bool(True)
        )
      )
    )
    replaceModule_or_Sequence(process, process.csc2DRecHits, process.csc2DRecHitsORG*process.csc2DRecHits)
  
    # mix recHits in DT
    print "Mixing DT recHit collection"
    process.dt1DRecHitsORG = process.dt1DRecHits.clone()
    process.dt1DRecHits = cms.EDProducer("DTRecHitMixer",
      recHitMuonDetCleanerConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("dt1DRecHitsORG", ""),
          cleanCollection1 = cms.bool(False),                                   
          collection2 = cms.InputTag("dt1DRecHits", "", inputProcess),
          cleanCollection2 = cms.bool(True)                         
        )
      )
    )
    replaceModule_or_Sequence(process, process.dt1DRecHits, process.dt1DRecHitsORG*process.dt1DRecHits)
  
    # mix recHits in RPC
    print "Mixing RPC recHit collection"
    process.rpcRecHitsORG = process.rpcRecHits.clone()
    process.rpcRecHits = cms.EDProducer("RPCRecHitMixer",
      recHitMuonDetCleanerConfig,
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("rpcRecHitsORG", ""),
          cleanCollection1 = cms.bool(False),                                  
          collection2 = cms.InputTag("rpcRecHits", "", inputProcess),
          cleanCollection2 = cms.bool(True)                      
        )
      )
    )
    replaceModule_or_Sequence(process, process.rpcRecHits, process.rpcRecHitsORG*process.rpcRecHits)

  if process.customization_options.muonMixingMode.value() == 2 or \
     process.customization_options.muonMixingMode.value() == 3:
    
    # CV: need to switch to coarse positions to prevent exception
    #     when running 'glbTrackQual' module on mixed globalMuon collection
    process.MuonTransientTrackingRecHitBuilderESProducerFromDisk = process.MuonTransientTrackingRecHitBuilderESProducer.clone(
      ComponentName = cms.string('MuonRecHitBuilderFromDisk'),
      ComputeCoarseLocalPositionFromDisk = cms.bool(True)
    )  
    process.ttrhbwrFromDisk = process.ttrhbwr.clone(
      ComponentName = cms.string('WithTrackAngleFromDisk'),
      ComputeCoarseLocalPositionFromDisk = cms.bool(True)
    )
    process.glbTrackQual.RefitterParameters.MuonRecHitBuilder = cms.string('MuonRecHitBuilderFromDisk')
    process.glbTrackQual.RefitterParameters.TrackerRecHitBuilder = cms.string('WithTrackAngleFromDisk')

    process.globalMuonsORG = process.globalMuons.clone()
    process.cleanedGlobalMuons = cms.EDProducer("GlobalMuonTrackCleaner",
      selectedMuons = process.customization_options.ZmumuCollection,
      tracks = cms.VInputTag("globalMuons"),
      dRmatch = cms.double(3.e-1),
      removeDuplicates = cms.bool(True),
      type = cms.string("links"),
      srcMuons = cms.InputTag("muons"),                                       
      verbosity = cms.int32(0)
    )
    process.globalMuons = cms.EDProducer("GlobalMuonTrackMixer",
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("globalMuonsORG", "", "EmbeddedRECO"),
          collection2 = cms.InputTag("cleanedGlobalMuons"),
        )
      ),
      verbosity = cms.int32(0) 
    )
    replaceModule_or_Sequence(process, process.globalMuons, process.cleanedGlobalMuons*process.globalMuonsORG*process.globalMuons)
  
    process.standAloneMuonsORG = process.standAloneMuons.clone()
    process.cleanedStandAloneMuons = process.cleanedGeneralTracks.clone(
      tracks = cms.VInputTag(
        cms.InputTag("standAloneMuons" ,""),
        cms.InputTag("standAloneMuons", "UpdatedAtVtx"),
      ),
      type = cms.string("outer tracks"),
      verbosity = cms.int32(0)
    )
    process.standAloneMuons = cms.EDProducer("TrackMixer",
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("standAloneMuonsORG", "", "EmbeddedRECO"),                                      
          collection2 = cms.InputTag("cleanedStandAloneMuons", "")
        ),                                             
        cms.PSet(
          collection1 = cms.InputTag("standAloneMuonsORG", "UpdatedAtVtx", "EmbeddedRECO"),                                       
          collection2 = cms.InputTag("cleanedStandAloneMuons", "UpdatedAtVtx")
        )
      ),
      verbosity = cms.int32(0) 
    )
    replaceModule_or_Sequence(process, process.standAloneMuons, process.cleanedStandAloneMuons*process.standAloneMuonsORG*process.standAloneMuons)

    process.tevMuonsORG = process.tevMuons.clone()
    if not process.customization_options.skipMuonDetRecHitMixing.value():
      process.tevMuonsORG.RefitterParameters.CSCRecSegmentLabel = cms.InputTag("csc2DRecHitsORG")
      process.tevMuonsORG.RefitterParameters.DTRecSegmentLabel = cms.InputTag("dt1DRecHitsORG")
      process.tevMuonsORG.RefitterParameters.RPCRecSegmentLabel = cms.InputTag("rpcRecHitsORG")
    process.tevMuonsORG.MuonCollectionLabel = cms.InputTag("globalMuonsORG")
    process.cleanedTeVMuons = cms.EDProducer("TeVMuonTrackCleaner",
      selectedMuons = process.customization_options.ZmumuCollection,
      tracks = cms.VInputTag(
        cms.InputTag("tevMuons", "default"),
        cms.InputTag("tevMuons", "dyt"),
        cms.InputTag("tevMuons", "firstHit"),
        cms.InputTag("tevMuons", "picky")      
      ),
      dRmatch = cms.double(3.e-1),
      removeDuplicates = cms.bool(True),
      type = cms.string("tev"),
      srcGlobalMuons_cleaned = cms.InputTag("cleanedGlobalMuons"),                                       
      verbosity = cms.int32(0)
    )
    process.tevMuons = cms.EDProducer("TeVMuonTrackMixer",
      todo = cms.VPSet(
        cms.PSet(
          collection1 = cms.InputTag("tevMuonsORG", "default", "EmbeddedRECO"),                                       
          collection2 = cms.InputTag("cleanedTeVMuons", "default")
        ),
        cms.PSet(
          collection1 = cms.InputTag("tevMuonsORG", "dyt", "EmbeddedRECO"),                                       
          collection2 = cms.InputTag("cleanedTeVMuons", "dyt")
        ),                                      
        cms.PSet(
          collection1 = cms.InputTag("tevMuonsORG", "firstHit", "EmbeddedRECO"),                                      
          collection2 = cms.InputTag("cleanedTeVMuons", "firstHit")
        ),                                             
        cms.PSet(
          collection1 = cms.InputTag("tevMuonsORG", "picky", "EmbeddedRECO"),                                       
          collection2 = cms.InputTag("cleanedTeVMuons", "picky")
        )                              
      ),
      srcGlobalMuons_cleaned = cms.InputTag("cleanedGlobalMuons"),                                    
      verbosity = cms.int32(0)                                    
    )
    replaceModule_or_Sequence(process, process.tevMuons, process.cleanedTeVMuons*process.tevMuonsORG*process.tevMuons)
    
  return process
