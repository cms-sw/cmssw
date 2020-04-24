import FWCore.ParameterSet.Config as cms

####
# Run the DT validation in local mode (additional histos activated), starting from a RECO input,
# optionally re-reconstructing segments from hits
#
# Configurable options:

reReco = False         # Set this to True to re-reconstruct hits
skipDeltaSuppr = False # Skip DRR (only when reReco=True)
step2FromDigi = False  # Remake step-2 hits instead of bypassing step-2; useful when re-fitting segments from hits since hits in 2D segments are otherwise non updated. 

# Optional DB for DT uncertainties, es.  "DTRecoUncertainties.db"
uncertDB = ""

doAngleCorr = False

#SAMPLE = "536"
#SAMPLE = "5312"
#SAMPLE = "620p6"
#SAMPLE = "700p4"
SAMPLE = "700p7"


####

process = cms.Process("DTValidationFromRECO")
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
        )

## Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


## Uncertainty DB
if uncertDB != "" : 
    process.GlobalTag.toGet = cms.VPSet(
        cms.PSet(record = cms.string("DTRecoUncertaintiesRcd"),
                 tag = cms.string("DTRecoUncertainties_test"),
                 connect = cms.untracked.string("sqlite_file:"+uncertDB))
        )



process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

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

# Set local mode
process.rechivalidation.doall = True
process.rechivalidation.local = True
process.seg4dvalidation.doall = True
process.seg4dvalidation.local = True

# Debug validation
#process.rechivalidation.debug = True
#process.seg4dvalidation.debug = True


process.options = cms.untracked.PSet(
    #FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000



if SAMPLE == "536" : 
    process.GlobalTag.globaltag = "START53_V7G::All"
    process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( 
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValZMM/GEN-SIM-RECO/v2/00000/08C1D822-F629-E211-A6B1-003048679188.root',
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValZMM/GEN-SIM-RECO/v2/00000/76156813-F529-E211-917B-003048678FA6.root',
                                ),
                            secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/v2/00000/EEEFF6D0-EC29-E211-94BB-003048678AC8.root',
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/v2/00000/C4AE2DAC-EB29-E211-8135-003048678BAC.root',
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/v2/00000/7A9F10B4-EB29-E211-88F1-003048FFCBA8.root',
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/v2/00000/64ECECBC-ED29-E211-AB98-002618943939.root'
                                )
        )

elif SAMPLE == "5312" : 
    process.GlobalTag.globaltag = "START53_LV2::All"
    process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_5_3_12_patch2/RelValZMM/GEN-SIM-RECO/START53_LV2-v1/00000/28D552C4-A82B-E311-952C-002590596486.root', 
       '/store/relval/CMSSW_5_3_12_patch2/RelValZMM/GEN-SIM-RECO/START53_LV2-v1/00000/EA8973F4-A92B-E311-A0A6-00261894396D.root', 
                                ),
                            secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_5_3_12_patch2/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2-v1/00000/1AC2E0AF-912B-E311-B9DE-003048679080.root',
       '/store/relval/CMSSW_5_3_12_patch2/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2-v1/00000/6247C454-912B-E311-BCCC-0025905822B6.root',
       '/store/relval/CMSSW_5_3_12_patch2/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2-v1/00000/CE2C4A01-9F2B-E311-8A2A-00261894395B.root',
       '/store/relval/CMSSW_5_3_12_patch2/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START53_LV2-v1/00000/F6429A09-9F2B-E311-A76F-003048FFCC18.root'
                                )
        )

elif SAMPLE == "620p6" : 
    process.GlobalTag.globaltag = "PRE_ST62_V6::All"
    process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(                                 
#       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValSingleMuPt100/GEN-SIM-RECO/PRE_ST62_V6-v1/00000/E4C71BBB-EDBE-E211-8CAF-002590593920.root'

       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-RECO/PRE_ST62_V6-v1/00000/1A1EDFF1-D5BE-E211-AE75-003048FFCB9E.root',
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-RECO/PRE_ST62_V6-v1/00000/3E430421-D9BE-E211-B2EB-0026189438A2.root'
                               ),
                           secondaryFileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/005D0C6A-D9BE-E211-A130-0026189438D6.root'
                              
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/4A3DC569-C9BE-E211-9B0B-003048678ED4.root',
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/5446A469-C3BE-E211-9E1C-00259059642E.root',
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/7025B923-D1BE-E211-8FE8-0026189438AA.root',
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/861065B9-C5BE-E211-B254-003048678A7E.root',
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/B085805B-C3BE-E211-9207-0026189437E8.root',
       '/store/relval/CMSSW_6_2_0_pre6_patch1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V6-v1/00000/C637F0BC-C5BE-E211-831A-00261894397B.root'

     )
    )

elif SAMPLE == "700p4" :
    process.GlobalTag.globaltag = "PRE_ST62_V8::All"    
    process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_0_0_pre4/RelValZMM/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/086DEE6A-1325-E311-BEB2-003048FFD752.root',
       '/store/relval/CMSSW_7_0_0_pre4/RelValZMM/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/A8B465D7-1025-E311-B656-003048D3C010.root'
                                ),
                            secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_0_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/04C8F739-0225-E311-B778-00261894385A.root',
       '/store/relval/CMSSW_7_0_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/429368D4-0125-E311-B80D-002618943882.root',
       '/store/relval/CMSSW_7_0_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/8E337AD2-FB24-E311-8208-00261894397E.root',
       '/store/relval/CMSSW_7_0_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/E47362B8-0025-E311-AA6C-003048FFCBA4.root'
                                )
    )

elif SAMPLE == "700p7" :
    #RelValZMM/CMSSW_7_0_0_pre7-PRE_ST62_V8-v2
    process.GlobalTag.globaltag = "PRE_ST62_V8::All"
    process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_0_0_pre7/RelValZMM/GEN-SIM-RECO/PRE_ST62_V8-v2/00000/1A18DD97-1946-E311-9689-003048FFD7A2.root',
       '/store/relval/CMSSW_7_0_0_pre7/RelValZMM/GEN-SIM-RECO/PRE_ST62_V8-v2/00000/7CEB59F4-0F46-E311-BE5C-0025905964B2.root'
                                ),
                            secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_7_0_0_pre7/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v2/00000/022794E1-0346-E311-8A49-0026189438FD.root',
       '/store/relval/CMSSW_7_0_0_pre7/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v2/00000/4C9B84CF-0246-E311-BBAD-003048FFCB84.root',
       '/store/relval/CMSSW_7_0_0_pre7/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v2/00000/BE8B527A-0E46-E311-92DC-0026189438F7.root',
       '/store/relval/CMSSW_7_0_0_pre7/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v2/00000/C65E52BB-FC45-E311-B207-00304867916E.root'
                                )

    )

process.source.inputCommands = cms.untracked.vstring("drop *",
                                                     "keep PSimHits_g4SimHits_MuonDTHits_SIM",
                                                     "keep DT*_*_*_*",
                                                     )

process.source.dropDescendantsOfDroppedBranches=cms.untracked.bool(False)


process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")


process.analysis = cms.Sequence(process.dtLocalRecoValidation_no2D)
process.clients = cms.Sequence(process.dtLocalRecoValidationClients)


### Skip DRR
if (skipDeltaSuppr) :
    process.dt4DSegments.Reco4DAlgoConfig.perform_delta_rejecting = False;
    process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.perform_delta_rejecting = False;

if (doAngleCorr) :
    process.dt4DSegments.Reco4DAlgoConfig.recAlgoConfig.doAngleCorr = True;
    process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.doAngleCorr = True; #Note: hit recomputation @step2 is not activated anyhow!

if (step2FromDigi) :
    process.dt4DSegments.Reco4DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True;
    process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True;


if (reReco) :
    #add  process.dt2DSegments if needed
#    process.jobPath = cms.Path(process.muonDTDigis*process.dtlocalreco+process.muonreco+process.dtLocalRecoAnal)
    process.jobPath = cms.Path(process.dt4DSegments+process.analysis + process.dqmSaver)

else :
    process.jobPath = cms.Path(process.analysis + process.dqmSaver)

