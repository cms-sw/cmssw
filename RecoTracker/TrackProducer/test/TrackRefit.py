import FWCore.ParameterSet.Config as cms
 
process = cms.Process("Refitting")

### standard MessageLoggerConfiguration
process.load("FWCore.MessageService.MessageLogger_cfi")

### Standard Configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
 

## Fitter-smoother: loosen outlier rejection as for first data-taking with LHC "collisions"
process.KFFittingSmootherWithOutliersRejectionAndRK.BreakTrajWith2ConsecutiveMissing = False
process.KFFittingSmootherWithOutliersRejectionAndRK.EstimateCut = 1000



### Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "IDEAL_V5::All"
process.GlobalTag.globaltag = 'GR09_P_V6::All'

### Track refitter specific stuff
process.load("RecoTracker.TrackProducer.TrackRefitters_cff") #the correct one
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff") #the one for backward compatibility


process.maxEvents = cms.untracked.PSet(     input = cms.untracked.int32(-1)     
)

process.source = cms.Source("PoolSource",
### tracks from collisions                            
    fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/c/chiochia/09_beam_commissioning/BSCskim_123151_Express.root') 
#'/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/1E04FC31-F99A-DD11-94EE-0018F3D096DE.root')

### tracks from cosmics                            
#    fileNames = cms.untracked.vstring(
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/005F51E5-0373-DD11-B6FA-001731AF6B7D.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/005F51E5-0373-DD11-B6FA-001731AF6B7D.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/006F3A6A-0373-DD11-A8E7-00304876A0FF.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/02CF5B1E-6476-DD11-A034-003048769E65.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/02DF31C3-A775-DD11-91C2-001A92971BB8.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/02F71F56-CE74-DD11-9DD0-001A92810AE4.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/0446C89C-E072-DD11-A341-0018F3D0960C.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/04750FC3-3E73-DD11-B054-00304876A147.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/04DFD531-0473-DD11-964E-0018F3D096AE.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/067111FB-3873-DD11-AD86-00304875A9C5.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/067982F4-E175-DD11-99F7-001731AF6AC5.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/0680EB9B-4F73-DD11-83F8-0018F3D0962E.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/06BF1AF3-E175-DD11-B467-00304876A147.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/0A3843F3-E175-DD11-8419-003048767EE7.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/0A5AAABA-3973-DD11-B949-003048767FA1.root',
#        '/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1/0000/0A911B18-0273-DD11-A5A6-001731A283E1.root')

### tracks from beam halo muons                            
) 

process.TRACKS = cms.OutputModule("PoolOutputModule",
                                outputCommands = cms.untracked.vstring('drop *_*_*_*', 
                                                                       'keep recoTracks_*_*_*',
                                                                       'keep recoTrackExtras_*_*_*',
                                                                       'keep TrackingRecHitsOwned_*_*_*'),

                                fileName = cms.untracked.string('refitting.root')
                                )

process.options = cms.untracked.PSet(     wantSummary = cms.untracked.bool(True) )


process.p1 = cms.Path(process.TrackRefitter
                      #process.TrackRefitterP5
                      #process.TrackRefitterBHM
)
process.outpath = cms.EndPath(process.TRACKS)

 
