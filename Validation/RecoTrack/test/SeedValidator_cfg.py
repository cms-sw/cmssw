import FWCore.ParameterSet.Config as cms

process = cms.Process("SEEDVALIDATOR")
process.load("Configuration/StandardSequences/GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP_31X::All'
#process.MessageLogger.categories = ['TrackAssociator', 'TrackValidator']
#process.MessageLogger.debugModules = ['*']
#process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('DEBUG'),
#    default = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackAssociator = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackValidator = cms.untracked.PSet(
#        limit = cms.untracked.int32(-1)
#    )
#)
#process.MessageLogger.cerr = cms.untracked.PSet(
#    placeholder = cms.untracked.bool(True)
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring([
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0007/DCDC08E0-514F-DE11-90B8-001D09F28755.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0006/867DB4EC-6D4E-DE11-B3CE-001D09F2503C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0006/78733E8B-674E-DE11-B5F8-001617C3B654.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0006/507FABB6-7C4E-DE11-BD2B-0019B9F72CE5.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0006/4EEFB307-744E-DE11-9E19-001D09F2A465.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0006/36E83BE1-814E-DE11-8C34-000423D9863C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0006/34BC1FBA-6B4E-DE11-ADE9-001D09F244BB.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/E69E523E-604F-DE11-BC91-003048678F26.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/D48299E4-294E-DE11-983F-00304875AA77.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/C0931F38-214E-DE11-ACA9-003048679164.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/BED51758-2D4E-DE11-A1D2-001731AF6A4B.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/92504E6F-224E-DE11-97E6-00304876A075.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/76083ABC-204E-DE11-BBF9-003048678ADA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/60DF2595-2C4E-DE11-B4B9-0018F3D0970C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/54A1A86E-2E4E-DE11-A1BA-0018F3D0966C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/24925847-2B4E-DE11-9557-001A92971BB8.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/102ED6C6-2D4E-DE11-BD7B-0018F3D096F0.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/STARTUP_31X_v1/0000/0C97089F-2C4E-DE11-85B8-0018F3D0966C.root' ] 

                                     ),
   secondaryFileNames=cms.untracked.vstring([
           '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/F20698E4-834E-DE11-AD06-001D09F2441B.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/ECCD5B61-874E-DE11-9485-001D09F250AF.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/E82E2C72-674E-DE11-9D8E-001617C3B706.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/E063E7B0-6D4E-DE11-A29E-001D09F253FC.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/BEC3F2B7-7C4E-DE11-96D4-001D09F2437B.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/A495AF09-6B4E-DE11-8E11-001D09F248F8.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/943F56B1-7D4E-DE11-90D1-001D09F2AF1E.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/92B7EF93-684E-DE11-A1F0-000423D6C8EE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/86936088-7E4E-DE11-9A2A-001D09F24EE3.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/7667DB33-674E-DE11-B5FF-001D09F29597.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/665E47B7-724E-DE11-B705-001D09F2AF96.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/60E23679-664E-DE11-A436-001D09F25442.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/58E8888D-704E-DE11-BD30-001D09F27003.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/50034FD3-6F4E-DE11-A525-001D09F23A34.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/44D03C92-6B4E-DE11-9EB7-001617C3B6FE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/44290C09-814E-DE11-9AB6-001D09F27003.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/32457AF8-784E-DE11-AD7A-001D09F23A07.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/10BDFCF9-6C4E-DE11-8A57-001D09F29619.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/10070AB8-754E-DE11-9B66-001D09F2514F.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/021AB859-6C4E-DE11-AAB2-001D09F253D4.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/FAC5D36B-224E-DE11-A638-0018F3D09688.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/F0AE1C5B-2E4E-DE11-87AD-0018F3D0960C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EEDA31B7-294E-DE11-8A0F-001A92971B62.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EEA77716-2B4E-DE11-AD91-0018F3D09642.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/ECBC6A61-204E-DE11-BBBB-003048679236.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EAE27C79-2C4E-DE11-8826-001A928116B0.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EA3FC178-2C4E-DE11-8D7C-0018F3D096E8.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/E4840369-2E4E-DE11-AAB7-0013D4C3BAFA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/E02F8963-204E-DE11-9C37-003048D15E02.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/DC16D029-214E-DE11-860D-0030486790BE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/D4F6E7AE-294E-DE11-8CAC-001BFCDBD160.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/CAB0D74C-2D4E-DE11-AA50-0018F3D09702.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/C42946B1-2D4E-DE11-9C3E-0018F3D095F6.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/ACED7BAC-2D4E-DE11-ACFB-0018F3D096E8.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/ACB4B857-2E4E-DE11-95D0-003048678B74.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/A8D4E4B7-2B4E-DE11-A6A1-0018F3D0966C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/98847871-2A4E-DE11-9874-001BFCDBD11E.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/94761BAF-1F4E-DE11-9155-003048678ADA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/90CE1802-214E-DE11-AE32-003048D15D22.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/8C04AA1B-2B4E-DE11-839D-001A9281171C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/8A915ABA-2D4E-DE11-9D43-001731AF692F.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/84710B49-2D4E-DE11-81E6-0018F3D095F8.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/80E67C4B-204E-DE11-A363-003048678FDE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/644402F8-5F4F-DE11-92A8-00304867926C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/506ECA80-224E-DE11-9647-0018F3D096EE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/40BA330F-2B4E-DE11-8F09-001BFCDBD100.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/3E09458C-224E-DE11-BF30-001A92971B36.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/2A8B4160-204E-DE11-B066-003048D15CC0.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/268A207A-2C4E-DE11-9015-001A92971ACE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/20FE4F45-2D4E-DE11-9CEC-001A928116B0.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/1282FB16-214E-DE11-9A8B-003048678FDE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/10662B7C-2C4E-DE11-8504-001A92971B80.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/0E56B627-214E-DE11-B804-001A92971BDA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/08133658-2D4E-DE11-B64B-0013D4C3BAFA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/06A6E5BC-2D4E-DE11-8958-001731AF6873.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/0038EF82-2C4E-DE11-B565-001731AF65F7.root' ])

)

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("Validation.RecoTrack.cuts_cff")

process.load("Validation.RecoTrack.TrackerSeedValidator_cff")
#process.multiTrackValidator.associators = cms.vstring('TrackAssociatorByHits','TrackAssociatorByChi2')
#process.multiTrackValidator.UseAssociators = True
#process.multiTrackValidator.label = ['cutsRecoTracks']
#process.multiTrackValidator.label_tp_effic = cms.InputTag("cutsTPEffic")
#process.multiTrackValidator.label_tp_fake  = cms.InputTag("cutsTPFake")
#process.multiTrackValidator.associatormap = cms.InputTag(assoc2GsfTracks)
process.trackerSeedValidator.outputFile = 'file.root'

# Tracking Truth and mixing module, if needed
#process.load("SimGeneral.MixingModule.mixNoPU_cfi")
#process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.ckftracks*process.cutsTPEffic*process.cutsTPFake*process.trackerSeedValidator)
#process.p = cms.Path(process.multiTrackValidator)
process.ep = cms.EndPath(process.evtInfo)


