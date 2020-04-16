import FWCore.ParameterSet.Config as cms

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
    process.caloParticles.allowDifferentSimHitProcesses = True
    process.mix.digitizers.caloParticles = process.caloParticles
    process.mix.digitizers.mergedtruth.ignoreTracksOutsideVolume = True
    process.mix.digitizers.mergedtruth.allowDifferentSimHitProcesses = True
    process.mix.digitizers.mergedtruth.select.signalOnlyTP = False

    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiStripDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiPixelDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedCaloTruth_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedTrackTruth_*')
    return process
 
def customize_step3(process):
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiStripDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_simSiPixelDigis_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedCaloTruth_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*_MergedTrackTruth_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoPFRecTracks_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoPFRecHits_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep recoGsfPFRecTracks_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_particleFlowBlock_*_*')
    return process
