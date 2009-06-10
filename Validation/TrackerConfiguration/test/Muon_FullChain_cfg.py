import FWCore.ParameterSet.Config as cms

# Put here the globaltag the file name and the number of events:

gtag=cms.string('IDEAL_31X::All')

inputfiles=cms.untracked.vstring(
[
       '/store/relval/CMSSW_3_1_0_pre5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/748D4BDD-B52B-DE11-90DF-000423D99614.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/A43647FD-0B2C-DE11-ADA0-000423D60FF6.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/EC0724B2-AC2B-DE11-BDB4-000423D991F0.root' ]
    )

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

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

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
process.tracks = cms.Sequence(process.offlineBeamSpot*process.recopixelvertexing*process.trackingParticles*process.trackingTruthValid*process.ckftracks*process.trackingRecHitsValid)
#process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)
process.p1 = cms.Path(process.simhits*process.mix*process.digis*process.rechits*process.tracks*process.trackinghits)
process.p1 = cms.Path(process.simhits*process.mix*process.digis*process.digitoraw*process.rawtodigi*process.rechits*process.tracks*process.trackinghits)
process.outpath = cms.EndPath(process.o1)
process.g4SimHits.Generator.HepMCProductLabel = 'source'


