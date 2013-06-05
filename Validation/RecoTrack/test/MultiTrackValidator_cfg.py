import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
)

# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/CE62D4D8-85ED-DE11-8BD2-000423D9853C.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/A0FB9B2E-85ED-DE11-8A8D-001D09F290CE.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/9A2F0DDF-85ED-DE11-B5D1-001D09F290CE.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/820C7C8C-86ED-DE11-83D4-001D09F295FB.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/685C77F0-87ED-DE11-A4A5-000423D60FF6.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/4CFCC894-86ED-DE11-B3F4-001D09F2447F.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/3EA206BD-B5ED-DE11-B481-000423D6C8E6.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/3CCCE28D-86ED-DE11-A583-000423D986C4.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/2CF90F4D-87ED-DE11-A3AF-003048D375AA.root' ] );


secFiles.extend( [
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/FA6E452B-85ED-DE11-AC27-001D09F25109.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/FA66FD3B-88ED-DE11-9A5D-001D09F28D4A.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/F2274DDB-85ED-DE11-9DED-003048D37580.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/D26001EE-87ED-DE11-BF98-000423D94494.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/D07779E8-87ED-DE11-9FE2-000423D98EC4.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/C6F3B22B-85ED-DE11-911C-001D09F24D67.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/BEAB94D5-85ED-DE11-9AEA-000423D6B444.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/BAB131B6-B5ED-DE11-A151-000423D6CA02.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/B8A44F23-85ED-DE11-92D7-001617E30E28.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/9637D444-87ED-DE11-9E21-000423D94534.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/8CFE0431-85ED-DE11-89B6-001D09F29597.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/86046CDD-85ED-DE11-AA2E-001617C3B654.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/704E4641-87ED-DE11-B6AC-000423D9A212.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/6271E489-86ED-DE11-BE7A-000423D99AAA.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/5C364B96-86ED-DE11-9140-0019B9F72BAA.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/54B8E090-86ED-DE11-A9B0-001D09F276CF.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/5087EC8C-86ED-DE11-BA63-001D09F2438A.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/42BFDFD9-85ED-DE11-BB7F-003048D375AA.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/2AE78E46-87ED-DE11-A50F-001D09F28755.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/10A1B68E-86ED-DE11-80C0-001D09F24DDF.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/0EC22640-87ED-DE11-855A-001D09F244BB.root',
       '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0004/041DBE8A-86ED-DE11-A83E-001D09F24FEC.root'] );

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(200) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP3X_V14::All'

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.GeometryPilot2_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

### validation-specific includes
#process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

########### configuration MultiTrackValidator ########
process.multiTrackValidator.outputFile = 'multitrackvalidator.root'
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
process.multiTrackValidator.skipHistoFit=cms.untracked.bool(False)
#process.cutsRecoTracks.quality = cms.vstring('','highPurity')
#process.cutsRecoTracks.quality = cms.vstring('')
process.multiTrackValidator.label = ['cutsRecoTracks']
process.multiTrackValidator.useLogPt=cms.untracked.bool(True)
process.multiTrackValidator.minpT = cms.double(0.1)
process.multiTrackValidator.maxpT = cms.double(3000.0)
process.multiTrackValidator.nintpT = cms.int32(40)
process.multiTrackValidator.UseAssociators = cms.bool(True)

#process.load("Validation.RecoTrack.cuts_cff")
#process.cutsRecoTracks.ptMin    = cms.double(0.5)
#process.cutsRecoTracks.minHit   = cms.int32(10)
#process.cutsRecoTracks.minRapidity  = cms.int32(-1.0)
#process.cutsRecoTracks.maxRapidity  = cms.int32(1.0)

process.quickTrackAssociatorByHits.useClusterTPAssociation = cms.bool(True)
process.load("SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi")

process.validation = cms.Sequence(
    process.tpClusterProducer *
    process.multiTrackValidator
)

# paths
process.p = cms.Path(
      process.cutsRecoTracks
    * process.validation
)
process.schedule = cms.Schedule(
      process.p
)


