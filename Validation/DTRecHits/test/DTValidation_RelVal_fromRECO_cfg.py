import FWCore.ParameterSet.Config as cms

process = cms.Process("DTValidationFromRECO")

## Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = "CRUZET4_V3P::All"
process.GlobalTag.globaltag = "IDEAL_31X::All"

process.load("Configuration.StandardSequences.MagneticField_cff")

#Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

# Validation RecHits
process.load("Validation.DTRecHits.DTRecHitQuality_cfi")
process.rechivalidation.doStep2 = False
# process.rechivalidation.recHitLabel = 'hltDt1DRecHits'
# process.rechivalidation.segment4DLabel = 'hltDt4DSegments'
# process.seg2dsuperphivalidation.segment4DLabel = 'hltDt4DSegments'
# process.seg4dvalidation.segment4DLabel = 'hltDt4DSegments'



process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
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
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0002/A6EBBA8D-1233-DE11-9CD8-001617C3B706.root',
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-RECO/IDEAL_31X_v1/0002/80322C09-1933-DE11-850A-000423D99BF2.root'
    ),
                            secondaryFileNames = cms.untracked.vstring( 
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/746C9E4E-D932-DE11-B1E6-001617DBCF90.root',
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/5C7FE942-1733-DE11-880D-001617C3B77C.root',
    '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/32F159D3-D832-DE11-9A86-000423D98A44.root'
    )

)


process.analysis = cms.Sequence(process.dtLocalRecoValidation_no2D)

process.p = cms.Path(process.analysis)

