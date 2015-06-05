import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
)


#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing"
    ,summaryOnly = cms.untracked.bool(True)
)

# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/067739D0-AFAB-E411-AC03-0025905A48D0.root'
                  ] )


secFiles.extend( [
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/041672CC-60AB-E411-B86C-003048FFCC0A.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/06DFB267-64AB-E411-A22D-0025905A60EE.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/0E932222-60AB-E411-952C-0025905A6088.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/1272643C-62AB-E411-969A-0025905A48B2.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/12BCC0BA-60AB-E411-83AA-0025905B85EE.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/2628B0FE-66AB-E411-83C3-0025905A48BC.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/26CA1B5B-61AB-E411-B134-0025905A612A.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/2CDDCDFD-66AB-E411-9ABA-0025905A60AA.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/32E319C0-60AB-E411-B983-0025905B855C.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/349C841F-63AB-E411-8805-0025905A48D8.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/36CBCF5E-61AB-E411-A75C-003048FF86CA.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/36E480F6-5EAB-E411-B272-0025905A48BC.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/4A703881-63AB-E411-A2BA-0025905B858A.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/4C0C75D7-5FAB-E411-9B34-0025905A608E.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/4E0352FF-66AB-E411-8751-0025905B8576.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/4E2491BA-64AB-E411-8F76-0025905B85D8.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/541FAFC2-60AB-E411-8B77-0025905A60A6.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/583B2424-60AB-E411-80B3-0025905B858E.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/58DBD23C-62AB-E411-8F4C-0025905A60EE.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/603F7120-60AB-E411-B0F6-0025905964C4.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/62ECB36D-71AB-E411-8BA6-003048FFD720.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/64BE7CF9-75AB-E411-A588-0025905B85E8.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/68815A64-61AB-E411-99D1-0025905A48BC.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/6EEC89BF-60AB-E411-9D75-0025905B85D8.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/76680CD8-5FAB-E411-99D8-0025905964A2.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/84CE7508-61AB-E411-9124-0025905B85EE.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/84FD8400-76AB-E411-8D13-0025905A612C.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/8670B486-5FAB-E411-A4B1-0025905A60B6.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/88BBBFD3-5FAB-E411-866C-0025905B85B2.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/88CF55D6-62AB-E411-924D-0025905A60B6.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/8A63EA7C-6EAB-E411-A961-0025905B85A2.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/8C717324-60AB-E411-B721-0025905AA9F0.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/9462F7B8-6CAB-E411-A0CC-0025905964B6.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/9E532222-60AB-E411-9353-0025905A60CE.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/A64050C2-60AB-E411-B9B9-0025905A60B0.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/AE05F253-65AB-E411-A528-0025905A60B6.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/AE8613EF-68AB-E411-9283-003048FFD744.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/BC55E7BE-60AB-E411-96A2-0025905A606A.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/CA18EDF3-5EAB-E411-BABF-002618943862.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/D6FF76A7-65AB-E411-8CA0-0025905B8610.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/DCB00F5C-61AB-E411-8A47-003048FFCB96.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/E295E1D2-5FAB-E411-8205-0026189438A9.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/E8821C20-6CAB-E411-8FC1-0025905A6126.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/F27EA4C0-60AB-E411-BD34-003048FFD744.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/F8D4970C-61AB-E411-B866-0025905A48D0.root',
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_MCRUN2_74_V1-v3/00000/FE40619E-61AB-E411-B453-0025905B858C.root',
        ] )
process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(400) )

### conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')


### validation-specific includes
#process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")
process.quickTrackAssociatorByHits.SimToRecoDenominator = 'reco'




########### configuration MultiTrackValidator ########
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
#process.cutsRecoTracks.quality = ['','highPurity']
#process.cutsRecoTracks.quality = ['']
process.multiTrackValidator.label = ['cutsRecoTracks']
process.multiTrackValidator.histoProducerAlgoBlock.useLogPt = True
process.multiTrackValidator.histoProducerAlgoBlock.minPt = 0.1
process.multiTrackValidator.histoProducerAlgoBlock.maxPt = 3000.0
process.multiTrackValidator.histoProducerAlgoBlock.nintPt = 40
process.multiTrackValidator.UseAssociators = True


#process.load("Validation.RecoTrack.cuts_cff")
#process.cutsRecoTracks.quality = ['highPurity']
#process.cutsRecoTracks.ptMin    = 0.5
#process.cutsRecoTracks.minHit   = 10
#process.cutsRecoTracks.minRapidity  = -1.0
#process.cutsRecoTracks.maxRapidity  = 1.0

process.quickTrackAssociatorByHits.useClusterTPAssociation = True
process.load("SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi")

process.validation = cms.Sequence(
    process.tpClusterProducer *
    process.quickTrackAssociatorByHits *
    process.multiTrackValidator
)

# paths
process.val = cms.Path(
      process.cutsRecoTracks
    * process.validation
)

# Output definition
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('file:MTV_inDQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


process.schedule = cms.Schedule(
      process.val,process.endjob_step,process.DQMoutput_step
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    wantSummary = cms.untracked.bool(True)
)



