import FWCore.ParameterSet.Config as cms

# Put here the globaltag the file name and the number of events:

gtag=cms.string('IDEAL_30X::All')

inputfiles=cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/1EF32A82-57E2-DD11-A475-000423D6B444.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/28804479-4BE2-DD11-A32D-000423D98EA8.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/44067402-4BE2-DD11-BD29-0030487C6090.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/6CCDD56F-4BE2-DD11-9078-001D09F27067.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/888637D0-4AE2-DD11-B5AD-000423D6CA6E.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/B6E15573-4BE2-DD11-B754-001D09F24E39.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/B8148F8C-4BE2-DD11-B0C8-001D09F28D54.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/D4855570-4BE2-DD11-8793-000423D98930.root')

nevents=cms.untracked.int32(1)

    
process = cms.Process("TrackerValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

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

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

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
process.digitoraw =cms.Sequence(process.siPixelRawData+process.SiStripDigiToRaw)
process.rawtodigi =cms.Sequence(process.siPixelDigis+process.SiStripRawToDigis)
process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.rechits = cms.Sequence(process.trackerlocalreco*process.trackerRecHitsValidation)
process.tracks = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.trackingParticles*process.trackingTruthValid*process.ckftracks*process.trackerRecHitsValidation)
process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)
process.p1 = cms.Path(process.simhits*process.mix*process.digis*process.rechits*process.tracks*process.trackinghits)
process.p1 = cms.Path(process.simhits*process.mix*process.digitoraw*process.rawtodigi*process.digis*process.rechits*process.tracks*process.trackinghits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'


