import FWCore.ParameterSet.Config as cms

def customize_ecalclustering_caloparticle(process):
    process.load('SimGeneral.MixingModule.caloTruthProducer_cfi')
    process.caloParticles.simHitCollections = cms.PSet(
        #hcal = cms.VInputTag(cms.InputTag('g4SimHits','HcalHits')),
        ecal = cms.VInputTag(
            cms.InputTag('g4SimHits','EcalHitsEE'),
            cms.InputTag('g4SimHits','EcalHitsEB'),
            cms.InputTag('g4SimHits','EcalHitsES'),
        )
    )
    process.caloParticles.doHGCAL = False
    process.caloParticles.allowDifferentSimHitProcesses = False
    process.mix.digitizers.caloParticles = process.caloParticles
    process.mix.digitizers.mergedtruth.ignoreTracksOutsideVolume = False
    process.mix.digitizers.mergedtruth.allowDifferentSimHitProcesses = False
    process.mix.digitizers.mergedtruth.select.signalOnlyTP = True

    process.PREMIXRAWoutput.outputCommands.append('keep *_*_MergedCaloTruth_*')
    return process

def customize_step2(process):
    process.load('SimGeneral.MixingModule.caloTruthProducer_cfi')
    process.caloParticles.simHitCollections = cms.PSet(
        hcal = cms.VInputTag(cms.InputTag('g4SimHits','HcalHits')),
        ecal = cms.VInputTag(
            cms.InputTag('g4SimHits','EcalHitsEE'),
            cms.InputTag('g4SimHits','EcalHitsEB'),
            cms.InputTag('g4SimHits','EcalHitsES'),
        )
    )
    process.caloParticles.doHGCAL = False
    process.mix.digitizers.caloParticles = process.caloParticles
    process.mix.digitizers.mergedtruth.ignoreTracksOutsideVolume = True
    process.mix.digitizers.mergedtruth.allowDifferentSimHitProcesses = False
    process.mix.digitizers.mergedtruth.select.signalOnlyTP = False

    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiStripDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiPixelDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedCaloTruth_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedTrackTruth_*')
    #process.FEVTDEBUGHLToutput.outputCommands.append("keep *_*G4*_*_*")
    #process.FEVTDEBUGHLToutput.outputCommands.append("keep SimClustersedmAssociation_mix_*_*")
    #process.FEVTDEBUGHLToutput.outputCommands.append("keep CaloParticlesedmAssociation_mix_*_*")
    return process
 
def customize_step3(process):
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiStripDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiPixelDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedCaloTruth_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedTrackTruth_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoPFRecTracks_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoPFRecHits_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoPFRecHitFractions_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoGsfPFRecTracks_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_particleFlowBlock_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoTracks_standAloneMuons_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoTrackExtras_standAloneMuons_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoMuons_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoTracks_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoGsfTracks_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoPFBlocks_*_*_*')
    #process.FEVTDEBUGHLToutput.outputCommands.append("keep *_*G4*_*_*")
    #process.FEVTDEBUGHLToutput.outputCommands.append("keep SimClustersedmAssociation_mix_*_*")
    #process.FEVTDEBUGHLToutput.outputCommands.append("keep CaloParticlesedmAssociation_mix_*_*")

    process.load("SimTracker.TrackerHitAssociation.tpClusterProducer_cfi")
    process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
    process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
    process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
      
    process.trackingParticleGsfTrackAssociation = process.trackingParticleRecoTrackAsssociation.clone(label_tr="electronGsfTracks")
    
    process.pfana = cms.EDAnalyzer('PFAnalysis')
    
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string("pfntuple.root")
    )
    
    process.pfana_path = cms.Path(
      process.tpClusterProducer*
      process.quickTrackAssociatorByHits*
      process.trackingParticleRecoTrackAsssociation*
      process.trackingParticleGsfTrackAssociation*
      process.pfana)
    
    process.schedule.append(process.pfana_path)

    # process.load("FWCore.MessageService.MessageLogger_cfi")
    # process.MessageLogger.cerr.threshold = "TRACE"
    # process.MessageLogger.debugModules = ["*"]


    return process
