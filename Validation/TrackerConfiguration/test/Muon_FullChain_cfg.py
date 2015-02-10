import FWCore.ParameterSet.Config as cms

# Put here the globaltag the file name and the number of events:

gtag=cms.string('IDEAL_31X::All')

inputfiles=cms.untracked.vstring(
['/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/B479D4A7-D14E-DE11-A78F-001D09F250AF.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/A6F81696-D04E-DE11-8B2E-001617C3B6DC.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/A625B90A-D34E-DE11-A4D3-000423D6A6F4.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/5441A448-D14E-DE11-B5BC-001D09F295A1.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/4C9657A5-D34E-DE11-9C9D-001D09F248F8.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/3EA6B3D7-514F-DE11-8DFD-000423D6A6F4.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/2C49163C-D24E-DE11-AA16-001D09F2438A.root',
       
'/store/relval/CMSSW_3_1_0_pre9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/1E69E467-D04E-DE11-8A75-001617E30D4A.root']
)

nevents=cms.untracked.int32(1)

    
process = cms.Process("TrackerValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = gtag


process.load("Configuration.StandardSequences.Services_cff")

process.load("SimG4Core.Configuration.SimG4Core_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Validation.TrackerHits.trackerHitsValidation_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Configuration.StandardSequences.Sim_cff")

process.load("Configuration.StandardSequences.Digi_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Validation.TrackerDigis.trackerDigisValidation_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

<<<<<<< Muon_FullChain_cfg.py
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
=======
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
>>>>>>> 1.4

process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")

process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")

process.load("Validation.RecoTrack.TrackValidation_cff")

process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")

process.maxEvents = cms.untracked.PSet(
    input = nevents
)
process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring('file:./Muon.root')
                            fileNames = inputfiles
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Muon_FullValidation.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")
process.trackerHitsValid.outputFile='TrackerHitHisto.root'
process.pixelDigisValid.outputFile='pixeldigihisto.root'
process.stripDigisValid.outputFile='stripdigihisto.root'
process.pixRecHitsValid.outputFile='pixelrechitshisto.root'
process.stripRecHitsValid.outputFile='sistriprechitshisto.root'
process.trackingTruthValid.outputFile='trackingtruthhisto.root'
process.multiTrackValidator.outputFile='validationPlots.root'
process.PixelTrackingRecHitsValid.outputFile='pixeltrackingrechitshist.root'
process.StripTrackingRecHitsValid.outputFile='striptrackingrechitshisto.root'

process.simhits = cms.Sequence(process.g4SimHits*process.trackerHitsValidation)
process.digitoraw =cms.Sequence(process.siPixelRawData*process.SiStripDigiToRaw*process.rawDataCollector)
process.rawtodigi =cms.Sequence(process.siPixelDigis+process.SiStripRawToDigis)
process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.rechits = cms.Sequence(process.trackerlocalreco*process.trackerRecHitsValidation)
process.tracks = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.trackingParticles*process.trackingTruthValid*process.ckftracks*process.trackingRecHitsValid)
#process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)

#process.p1 = cms.Path(process.simhits*process.mix*process.digis*process.rechits*process.tracks*process.trackinghits)
process.p1 = cms.Path(process.simhits*process.mix*process.digitoraw*process.rawtodigi*process.digis*process.rechits*process.tracks)

process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'


