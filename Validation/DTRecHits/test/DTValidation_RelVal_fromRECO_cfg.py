import FWCore.ParameterSet.Config as cms

process = cms.Process("DTValidationFromRECO")

## Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = "CRUZET4_V3P::All"
process.GlobalTag.globaltag = "MC_31X_V1::All"

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# DQM services
process.load("DQMServices.Core.DQMStore_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = 'Offline'
# FIXME: correct this
process.dqmSaver.workflow = '/Cosmics/CMSSW_2_2_X-Testing/RECO'

# Validation RecHits
process.load("Validation.DTRecHits.DTRecHitQuality_cfi")
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
    #TryToContinue = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( 
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0008/BADD3EA6-0458-DE11-B820-001D09F23944.root',
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0008/26D81C3A-7857-DE11-9D02-001D09F29849.root'
'/store/relval/CMSSW_3_1_0_pre11/RelValSingleMuPt100/GEN-SIM-RECO/MC_31X_V1-v1/0000/A632041C-A964-DE11-B489-0030487A3232.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValSingleMuPt100/GEN-SIM-RECO/MC_31X_V1-v1/0000/4CBC3EFA-EC64-DE11-877E-000423D987FC.root'

    ),
                            secondaryFileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E2CF66DF-7557-DE11-8F38-001D09F25325.root',
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/86023B5A-6E57-DE11-974A-000423D6C8E6.root',
#    '/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4403B0A3-0458-DE11-A53F-001D09F241D2.root'
'/store/relval/CMSSW_3_1_0_pre11/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/888F0E57-ED64-DE11-B50E-001617DC1F70.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/50FB5A53-A864-DE11-A157-001D09F24682.root',
        '/store/relval/CMSSW_3_1_0_pre11/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0000/0628442A-7464-DE11-8D2F-001D09F25217.root'

    )

)


process.analysis = cms.Sequence(process.dtLocalRecoValidation_no2D)
process.clients = cms.Sequence(process.dtLocalRecoValidationClients)
process.p = cms.Path(process.analysis + process.clients + process.dqmSaver)

