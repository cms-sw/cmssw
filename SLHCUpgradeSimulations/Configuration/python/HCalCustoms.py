import FWCore.ParameterSet.Config as cms

def customise_HcalPhase0(process):
    process.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")

    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):    
        process.mix.digitizers.hcal.HcalReLabel.RelabelHits=cms.untracked.bool(True)

    process.es_hardcode.HEreCalibCutoff = cms.double(20.) #for aging

    process.es_hardcode.toGet = cms.untracked.vstring(
        'GainWidths',
        'RespCorrs'
        )

    if hasattr(process,'g4SimHits'):
        process=customise_Sim(process)
    if hasattr(process,'validation_step'):
        process=customise_ValidationPhase0(process)
		
    return process

def customise_HcalPhase1(process):
    #common stuff
    process.load("CalibCalorimetry/HcalPlugins/Hcal_Conditions_forGlobalTag_cff")
    process.es_hardcode.toGet = cms.untracked.vstring(
                'GainWidths',
                'MCParams',
                'RecoParams',
                'RespCorrs',
                'QIEData',
                'Gains',
                'Pedestals',
                'PedestalWidths',
                'ChannelQuality',
                'ZSThresholds',
                'TimeCorrs',
                'LUTCorrs',
                'LutMetadata',
                'L1TriggerObjects',
                'PFCorrs',
                'ElectronicsMap',
                'CholeskyMatrices',
                'CovarianceMatrices'
                )

    # Special Upgrade trick (if absent - regular case assumed)
    process.es_hardcode.GainWidthsForTrigPrims = cms.bool(True)
    process.es_hardcode.HEreCalibCutoff = cms.double(100.) #for aging
    
    if hasattr(process,'g4SimHits'):
        process=customise_Sim(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'reconstruction_step'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_ValidationPhase1(process)
    process=customise_condOverRides(process)
    return process

def customise_HcalPhase2(process):
    process = customise_HcalPhase1(process)
    if hasattr(process,'digitisation_step') and hasattr(process, 'mix'):

        # these are the new sampling factors,  they reuse the old ones for
        # ieta < 21.  For ieta greater than 21 it is using the function
        # samplingFraction = 188.441 + 0.834*eta
        # eta is the highest eta broundary of the ieta.  This is currently
        # taken for HE from ieta 16 to 33 inclusive.  Which would extend to
        # an eta of 3.0.  For the option going to 4.0 it is unclear how many
        # ieta's there will be from 3 to 4, but this vector would need to be
        # extended.
        newFactors = cms.vdouble(
            210.55, 197.93, 186.12, 189.64, 189.63,
            189.96, 190.03, 190.11, 190.18, 190.25,
            190.32, 190.40, 190.47, 190.54, 190.61,
            190.69, 190.83, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94,
            190.94, 190.94, 190.94, 190.94, 190.94)
        process.mix.digitizers.hcal.he.samplingFactors = newFactors
        process.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*len(newFactors))

    if hasattr(process,'reconstruction_step'):
        process.towerMaker.HcalPhase = cms.int32(2)
        process.towerMakerPF.HcalPhase = cms.int32(2)
        process.towerMakerWithHO.HcalPhase = cms.int32(2)
        process.CaloTowerConstituentsMapBuilder.MapFile = cms.untracked.string("")

    return process

def customise_Sim(process):
    process.g4SimHits.HCalSD.TestNumberingScheme = True

    return process

def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.hcalRawData)
        
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.hcalDigis)

    return process

def customise_Digi(process):
    if hasattr(process,'mix'):
        process.mix.digitizers.hcal.HBHEUpgradeQIE = True
        process.mix.digitizers.hcal.hb.siPMCells = cms.vint32([1])
        process.mix.digitizers.hcal.hb.photoelectronsToAnalog = cms.vdouble([10.]*16)
        process.mix.digitizers.hcal.hb.pixels = cms.int32(4500*4*2)
        process.mix.digitizers.hcal.he.photoelectronsToAnalog = cms.vdouble([10.]*16)
        process.mix.digitizers.hcal.he.pixels = cms.int32(4500*4*2)
        process.mix.digitizers.hcal.HFUpgradeQIE = True
        process.mix.digitizers.hcal.HcalReLabel.RelabelHits=cms.untracked.bool(True)
        process.mix.digitizers.hcal.doTimeSlew = False
    if hasattr(process,'simHcalDigis'):
        process.simHcalDigis.useConfigZSvalues=cms.int32(1)
        process.simHcalDigis.HBlevel=cms.int32(16)
        process.simHcalDigis.HElevel=cms.int32(16)
        process.simHcalDigis.HOlevel=cms.int32(8)
        process.simHcalDigis.HFlevel=cms.int32(10)

    process.digitisation_step.remove(process.simHcalTriggerPrimitiveDigis)
    process.digitisation_step.remove(process.simHcalTTPDigis)
    
    return process

def customise_Reco(process):
    #--- CaloTowers maker input customization
    process.towerMaker.hfInput = cms.InputTag("hfUpgradeReco")
    process.towerMaker.hbheInput = cms.InputTag("hbheUpgradeReco") 
    process.towerMakerPF.hfInput = cms.InputTag("hfUpgradeReco")
    process.towerMakerPF.hbheInput = cms.InputTag("hbheUpgradeReco") 
    process.towerMakerWithHO.hfInput = cms.InputTag("hfUpgradeReco")
    process.towerMakerWithHO.hbheInput = cms.InputTag("hbheUpgradeReco") 
    process.towerMaker.HcalPhase = cms.int32(1)
    process.towerMakerPF.HcalPhase = cms.int32(1)
    process.towerMakerWithHO.HcalPhase = cms.int32(1)
    process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbheUpgradeReco")
    process.particleFlowRecHitHCAL.hcalRecHitsHF = cms.InputTag("hfUpgradeReco")
    process.ak5JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.ak5JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.ak7JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.ak7JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.ca4JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.ca4JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.ca6JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.ca6JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.gk5JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.gk5JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.gk7JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.gk7JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.ic5JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.ic5JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.ic7JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.ic7JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.kt4JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.kt4JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.kt6JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.kt6JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.sc5JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.sc5JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.sc7JetID.hfRecHitsColl = cms.InputTag("hfUpgradeReco")
    process.sc7JetID.hbheRecHitsColl = cms.InputTag("hbheUpgradeReco")
    process.hfEMClusters.hits = cms.InputTag("hfUpgradeReco")
    
    process.muons1stStep.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muons1stStep.CaloExtractorPSet.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muons1stStep.JetExtractorPSet.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")

    process.muonsFromCosmics.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muonsFromCosmics.CaloExtractorPSet.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muonsFromCosmics.JetExtractorPSet.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muonsFromCosmics1Leg.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muonsFromCosmics1Leg.CaloExtractorPSet.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
    process.muonsFromCosmics1Leg.JetExtractorPSet.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")

    process.interestingTrackEcalDetIds.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")

    process.hcalnoise.recHitCollName=cms.string('hbheUpgradeReco')
    process.reducedHcalRecHits.hfTag=cms.InputTag("hfUpgradeReco")
    process.reducedHcalRecHits.hbheTag=cms.InputTag("hbheUpgradeReco")

    process.load("RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.HFUpgradeReconstructor_cfi")

    process.reconstruction_step.replace(process.hfreco,process.hfUpgradeReco)
    process.reconstruction_step.remove(process.hbhereco)
    process.reconstruction_step.replace(process.hbheprereco,process.hbheUpgradeReco)

    process.horeco.digiLabel = "simHcalDigis" 
    process.hbheUpgradeReco.digiLabel = cms.InputTag("simHcalDigis","HBHEUpgradeDigiCollection")
    process.hfUpgradeReco.digiLabel = cms.InputTag("simHcalDigis","HFUpgradeDigiCollection")
    process.hbheUpgradeReco.correctForTimeslew = False

    process.zdcreco.digiLabel = "simHcalUnsuppressedDigis"
    process.hcalnoise.digiCollName=cms.string('simHcalDigis')
    if hasattr(process, 'hcalnoise'):
        process.reconstruction_step.remove(process.hcalnoise)
        # process.reconstruction_step.highlevelreco.metrecoPlusHCALNoise.remove(process.hcalnoise)
        

    # not sure why these are missing - but need to investigate later
    process.reconstruction_step.remove(process.castorreco)
    process.reconstruction_step.remove(process.CastorTowerReco)
    process.reconstruction_step.remove(process.ak7BasicJets)
    process.reconstruction_step.remove(process.ak7CastorJetID)

    return process

def customise_DQM(process):
    process.dqmoffline_step.remove(process.hcalDigiMonitor)
    process.dqmoffline_step.remove(process.hcalDeadCellMonitor)
    process.dqmoffline_step.remove(process.hcalBeamMonitor)
    process.dqmoffline_step.remove(process.hcalRecHitMonitor)
    process.dqmoffline_step.remove(process.hcalDetDiagNoiseMonitor)
    process.dqmoffline_step.remove(process.hcalNoiseMonitor)
    process.dqmoffline_step.remove(process.RecHitsDQMOffline)
    process.dqmoffline_step.remove(process.zdcMonitor)
    process.dqmoffline_step.remove(process.hcalMonitor)
    process.dqmoffline_step.remove(process.hcalHotCellMonitor)
    process.dqmoffline_step.remove(process.hcalRawDataMonitor)
    process.ExoticaDQM.JetIDParams.hbheRecHitsColl=cms.InputTag("hbheUpgradeReco")
    process.ExoticaDQM.JetIDParams.hfRecHitsColl=cms.InputTag("hfUpgradeReco")

    if hasattr(process, 'NoiseRatesDQMOffline'):
        process.dqmoffline_step.remove(process.NoiseRatesDQMOffline)
    if hasattr(process, 'HBHENoiseFilterResultProducer'):
        process.dqmoffline_step.remove(process.HBHENoiseFilterResultProducer)
    if hasattr(process, 'towerSchemeBAnalyzer'):
        process.dqmoffline_step.remove(process.towerSchemeBAnalyzer)
        
    return process

def customise_harvesting(process):
    process.hcaldigisClient.doSLHC    = cms.untracked.bool(True)
    process.hcalrechitsClient.doSLHC  = cms.untracked.bool(True)
    return process

def customise_ValidationPhase0(process):
#    process.AllHcalDigisValidation.doSLHC = cms.untracked.bool(True)
    process.AllHcalDigisValidation.digiLabel = cms.InputTag("simHcalDigis")
    process.validation_step.remove(process.globalhitsanalyze)
    return process

def customise_ValidationPhase1(process):
    process.AllHcalDigisValidation.doSLHC = cms.untracked.bool(True)
    process.AllHcalDigisValidation.digiLabel = cms.InputTag("simHcalDigis")
    process.RecHitsValidation.doSLHC = cms.untracked.bool(True)
    process.RecHitsValidation.HBHERecHitCollectionLabel = cms.untracked.InputTag("hbheUpgradeReco")
    process.RecHitsValidation.HFRecHitCollectionLabel = cms.untracked.InputTag("hfUpgradeReco") 
    process.validation_step.remove(process.globalhitsanalyze)

    if hasattr(process, 'NoiseRatesValidation'):
        # process.validation_step.hcalRecHitsValidationSequence.remove(process.NoiseRatesValidation)
        process.validation_step.remove(process.NoiseRatesValidation)

    return process

def customise_condOverRides(process):
    return process
