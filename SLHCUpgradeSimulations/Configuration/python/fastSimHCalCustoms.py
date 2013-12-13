import FWCore.ParameterSet.Config as cms

def customise_fastHcalPhase1(process):
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
    
    process.es_hardcode.hcalTopologyConstants.mode=cms.string('HcalTopologyMode::SLHC')
    process.es_hardcode.hcalTopologyConstants.maxDepthHB=cms.int32(3)
    process.es_hardcode.hcalTopologyConstants.maxDepthHE=cms.int32(5)
    process.es_hardcode.HcalReLabel.RelabelHits=cms.untracked.bool(True)
    # Special Upgrade trick (if absent - regular case assumed)
    process.es_hardcode.GainWidthsForTrigPrims = cms.bool(True)
    process.es_hardcode.HEreCalibCutoff = cms.double(100.) #for aging
    
    process.hcalTopologyIdeal.hcalTopologyConstants.mode=cms.string('HcalTopologyMode::SLHC')
    process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHB=cms.int32(3)
    process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHE=cms.int32(5)
    process.HcalHardcodeGeometryEP.HcalReLabel.RelabelHits=cms.untracked.bool(False)
    

    if hasattr(process,'famosSimHits'):
        process=customise_Sim(process)
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'caloDigis'):
        process=customise_Digi(process)
    if hasattr(process,'hcalRecHitSequence'):
        process=customise_Reco(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
##    if hasattr(process,'validation_step'):
##        process=customise_Validation(process)
    process=customise_condOverRides(process)
    return process


def customise_Sim(process):
##    process.g4SimHits.HCalSD.TestNumberingScheme = False

    return process

def customise_DigiToRaw(process):
    process.caloDigis.remove(process.hcalRawData)

    return process

def customise_RawToDigi(process):
    process.caloDigis.remove(process.hcalDigis)

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
        process.mix.digitizers.hcal.HcalReLabel.RelabelHits=cms.untracked.bool(False)

    if hasattr(process,'HcalTPGCoderULUT'):
        process.HcalTPGCoderULUT.hcalTopologyConstants.mode=cms.string('HcalTopologyMode::SLHC')
        process.HcalTPGCoderULUT.hcalTopologyConstants.maxDepthHB=cms.int32(3)
        process.HcalTPGCoderULUT.hcalTopologyConstants.maxDepthHE=cms.int32(5)

    if hasattr(process,'simHcalDigis'):
        process.simHcalDigis.useConfigZSvalues=cms.int32(1)
        process.simHcalDigis.HBlevel=cms.int32(16)
        process.simHcalDigis.HElevel=cms.int32(16)
        process.simHcalDigis.HOlevel=cms.int32(8)
        process.simHcalDigis.HFlevel=cms.int32(10)

##    process.caloDigis.remove(process.simHcalTriggerPrimitiveDigis)
##    process.caloDigis.remove(process.simHcalTTPDigis)

    return process

def customise_Reco(process):
    #--- CaloTowers maker input customization
    process.towerMaker.hfInput = cms.InputTag("hfUpgradeReco")
    process.towerMaker.hbheInput = cms.InputTag("hbheUpgradeReco") 
    process.towerMakerPF.hfInput = cms.InputTag("hfUpgradeReco")
    process.towerMakerPF.hbheInput = cms.InputTag("hbheUpgradeReco") 
    process.towerMakerWithHO.hfInput = cms.InputTag("hfUpgradeReco")
    process.towerMakerWithHO.hbheInput = cms.InputTag("hbheUpgradeReco") 
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

##    process.muonsFromCosmics.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
##    process.muonsFromCosmics.CaloExtractorPSet.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
##    process.muonsFromCosmics.JetExtractorPSet.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
##    process.muonsFromCosmics1Leg.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
##    process.muonsFromCosmics1Leg.CaloExtractorPSet.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")
##    process.muonsFromCosmics1Leg.JetExtractorPSet.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")

    process.interestingTrackEcalDetIds.TrackAssociatorParameters.HBHERecHitCollectionLabel=cms.InputTag("hbheUpgradeReco")

    process.hcalnoise.recHitCollName=cms.string('hbheUpgradeReco')
    process.reducedHcalRecHits.hfTag=cms.InputTag("hfUpgradeReco")
    process.reducedHcalRecHits.hbheTag=cms.InputTag("hbheUpgradeReco")

    process.load("RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.HFUpgradeReconstructor_cfi")

    process.hcalRecHitSequence.replace(process.hfreco,process.hfUpgradeReco)
    process.hcalRecHitSequence.replace(process.hbhereco,process.hbheUpgradeReco)
    process.hcalRecHitSequence.replace(process.hbheprereco,process.hbheUpgradeReco)

    process.horeco.digiLabel = "simHcalDigis" 
    process.hbheUpgradeReco.digiLabel = cms.InputTag("simHcalDigis","HBHEUpgradeDigiCollection")
    process.hfUpgradeReco.digiLabel = cms.InputTag("simHcalDigis","HFUpgradeDigiCollection")

##    process.zdcreco.digiLabel = "simHcalUnsuppressedDigis"
    process.hcalnoise.digiCollName=cms.string('simHcalDigis')

    # not sure why these are missing - but need to investigate later
##    process.hcalRecHitSequence.remove(process.castorreco)
##    process.hcalRecHitSequence.remove(process.CastorTowerReco)
##    process.hcalRecHitSequence.remove(process.ak7BasicJets)
##    process.hcalRecHitSequence.remove(process.ak7CastorJetID)

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
    return process

def customise_harvesting(process):
    return process

def customise_Validation(process):
    process.validation_step.remove(process.AllHcalDigisValidation)
    process.validation_step.remove(process.RecHitsValidation)
    process.validation_step.remove(process.globalhitsanalyze)
    return process

def customise_condOverRides(process):
    return process
