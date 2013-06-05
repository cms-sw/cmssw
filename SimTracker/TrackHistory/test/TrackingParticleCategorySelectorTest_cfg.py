import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackCategorySelectorCascades")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.load("SimTracker.TrackHistory.Playback_cff")
process.load("SimTracker.TrackHistory.TrackClassifier_cff")

from SimTracker.TrackHistory.CategorySelectors_cff import * 

process.trackSelector = TrackingParticleCategorySelector(
    src = cms.InputTag('mix', 'MergedTrackTruth'),
    cut = cms.string("is('XiDecay') || is('OmegaDecay')")
)

process.trackSelector.filter = cms.bool(True)

process.GlobalTag.globaltag = 'MC_38Y_V9::All'

process.p = cms.Path(process.playback * process.trackSelector)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0020/AA5D39CB-33B6-DF11-8776-001A92810AA2.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0020/9A37DC35-1BB6-DF11-9949-003048679162.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0020/784A1867-49B6-DF11-B85E-0018F3D09700.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0020/560379A8-29B6-DF11-9ED6-00304867C026.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0020/44BC199D-24B6-DF11-93B9-003048679168.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0020/32DF6ADB-15B6-DF11-9247-001A928116DE.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0019/F02EB15D-0DB6-DF11-9AEC-00261894387A.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0019/883C0AD1-04B6-DF11-AAF2-0026189438E9.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0019/2284E8D0-FEB5-DF11-8029-002618943945.root' ] );

secFiles.extend( [
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/FCF778C5-21B6-DF11-8686-00261894395C.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/F075DC4C-1AB6-DF11-B548-003048678FF4.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/EEFDD629-26B6-DF11-ABA0-0018F3D096B4.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/E85410E7-15B6-DF11-81F5-003048D15DCA.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/E676EA20-28B6-DF11-907B-003048678FF4.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/E05C93A3-30B6-DF11-B701-00261894385D.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/BA4E6136-2CB6-DF11-9725-00304867905A.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/A682593C-22B6-DF11-8BB1-001A92971B36.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/9627A8AB-33B6-DF11-97A8-001A928116AE.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/90A0BDF7-3BB6-DF11-9C3B-0030486790B8.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/8047858D-1AB6-DF11-913D-003048679046.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/12B8D560-49B6-DF11-AE3B-002618943843.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/FC0E62CE-FFB5-DF11-98FD-002618943905.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/E6385FC5-0BB6-DF11-9F0D-003048678B0A.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/B28E775C-FEB5-DF11-AF63-002618943963.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/AE2EBCC1-0DB6-DF11-81FC-003048678B1C.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/AC808849-0AB6-DF11-AA31-002618943983.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/A4B1683A-10B6-DF11-BD22-001A92971BD8.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/50984DCF-FEB5-DF11-B4A8-0030486792AC.root',
       '/store/relval/CMSSW_3_9_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/38DC3DCC-03B6-DF11-B9FC-001A928116F2.root' ] );

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('selectedcascadesevents.root'),
    SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('p')
    )
)
process.e = cms.EndPath(process.out)



