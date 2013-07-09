import FWCore.ParameterSet.Config as cms

process = cms.Process("DTValidationFromRECO")

## Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = "CRUZET4_V3P::All"
process.GlobalTag.globaltag = "MC_38Y_V12::All"

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

# DQM services
process.load("DQMServices.Core.DQMStore_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = 'Offline'
# FIXME: correct this
process.dqmSaver.workflow = '/Cosmics/CMSSW_2_2_X-Testing/RECO'

# Validation RecHits
process.load("Validation.DTRecHits.DTRecHitQualityAll_cfi")
process.load("Validation.DTRecHits.DTRecHitClients_cfi")
##process.rechivalidation.doStep2 = False
# process.rechivalidation.recHitLabel = 'hltDt1DRecHits'
# process.rechivalidation.segment4DLabel = 'hltDt4DSegments'
# process.seg2dsuperphivalidation.segment4DLabel = 'hltDt4DSegments'
# process.seg4dvalidation.segment4DLabel = 'hltDt4DSegments'



process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

process.options = cms.untracked.PSet(
    #FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')
                                                              ),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( 
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0008/BADD3EA6-0458-DE11-B820-001D09F23944.root',
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0008/26D81C3A-7857-DE11-9D02-001D09F29849.root'
        '/store/relval/CMSSW_3_8_5/RelValSingleMuPt100/GEN-SIM-RECO/MC_38Y_V12-v1/0039/EAD3AB7B-CBD1-DF11-843E-0026189438E7.root',
        '/store/relval/CMSSW_3_8_5/RelValSingleMuPt100/GEN-SIM-RECO/MC_38Y_V12-v1/0039/62500BFE-CCD1-DF11-A866-002618943870.root'

    ),
                            secondaryFileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E2CF66DF-7557-DE11-8F38-001D09F25325.root',
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/86023B5A-6E57-DE11-974A-000423D6C8E6.root',
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4403B0A3-0458-DE11-A53F-001D09F241D2.root'
       '/store/relval/CMSSW_3_8_5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V12-v1/0043/8C1BC461-E7D2-DF11-B643-0026189437F0.root',
        '/store/relval/CMSSW_3_8_5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V12-v1/0039/540EEAFE-C9D1-DF11-A6B6-002354EF3BD0.root',
        '/store/relval/CMSSW_3_8_5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V12-v1/0039/327B6479-CBD1-DF11-BDB8-0026189438DB.root',
        '/store/relval/CMSSW_3_8_5/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V12-v1/0039/1A4F3DF8-CBD1-DF11-A2FE-003048678E80.root'

    )

)


process.analysis = cms.Sequence(process.dtLocalRecoValidation_no2D)
process.clients = cms.Sequence(process.dtLocalRecoValidationClients)
process.p = cms.Path(process.analysis + process.dqmSaver)

