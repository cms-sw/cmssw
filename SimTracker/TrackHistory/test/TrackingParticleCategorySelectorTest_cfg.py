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

#process.trackSelector = TrackCategorySelector( 
#    src = cms.InputTag('generalTracks'),
#    cut = cms.string("is('BWeakDecay')")
#)

#process.trackSelector = TrackCategorySelector(
#    src = cms.InputTag('generalTracks'),
#    cut = cms.string("is('Xi') || is('Omega')")
#)

process.trackSelector = TrackingParticleCategorySelector(
    src = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    cut = cms.string("is('Xi') || is('Omega')")
)

process.trackSelector.filter = cms.bool(True)

#process.trackHistoryAnalyzer = cms.EDAnalyzer("TrackHistoryAnalyzer",
#    process.trackClassifier
#)

#process.trackHistoryAnalyzer.trackProducer = 'trackSelector'

process.GlobalTag.globaltag = 'MC_31X_V1::All'

process.p = cms.Path(process.playback * process.trackSelector)
#process.p = cms.Path(process.trackHistoryAnalyzer)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( (
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/CE80AD93-AD64-DE11-B461-001D09F2910A.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/6474DFFC-7964-DE11-BCEC-001D09F2426D.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/5E18C473-AD64-DE11-B659-001D09F29524.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/56E466FA-EC64-DE11-9A31-0019B9F581C9.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/5057E9DA-5E64-DE11-9FB4-000423D99BF2.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/360942E6-9464-DE11-9064-00304879FA4A.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0000/06B63869-A864-DE11-9B41-001D09F28E80.root'
) );

secFiles.extend( (
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/EE79E0FA-9064-DE11-8330-001D09F25401.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/DCCAF16F-AD64-DE11-A6BC-001D09F2545B.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/CAE5DCE9-A864-DE11-8623-000423D9A212.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/C67C3F79-7464-DE11-B0A2-001D09F2503C.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/C02442DF-9664-DE11-A8CB-001617C3B6DC.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/B61E5707-B064-DE11-A573-001D09F29321.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/AE7DC67D-9B64-DE11-8914-001D09F28EA3.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/A8EE4892-AD64-DE11-83B7-001D09F23A02.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/90A98597-5F64-DE11-875A-001D09F242EA.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/8E17CFE8-AD64-DE11-9977-000423D98BE8.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/8CB13B8F-AD64-DE11-BB8A-001D09F2AF1E.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/8206A7B4-4A64-DE11-B026-001D09F291D2.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/78C6BF4E-AA64-DE11-A5E4-0019B9F72D71.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/668A2733-ED64-DE11-9A74-001D09F23A20.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/64ADDB99-7064-DE11-A877-001D09F241B9.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/5C08A662-AD64-DE11-B209-001D09F23A02.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/468321E6-7B64-DE11-8F06-001D09F2983F.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/446D92B6-5764-DE11-9808-001D09F242EA.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/304F9977-AD64-DE11-B462-001D09F34488.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/2E10C542-A764-DE11-8E42-001D09F27003.root',
       '/store/relval/CMSSW_3_1_0_pre11/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/0AA8D1D4-8564-DE11-87F6-001D09F29849.root' 
) )

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('selectedcascadesevents.root'),
    SelectEvents = cms.untracked.PSet(
                SelectEvents = cms.vstring('p')
    )
)
process.e = cms.EndPath(process.out)



