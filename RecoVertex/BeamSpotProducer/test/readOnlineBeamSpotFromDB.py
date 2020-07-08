import FWCore.ParameterSet.Config as cms

process = cms.Process("readDB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")


""" 
process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(


cms.PSet(
    record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
    tag = cms.string("BSOnLineLegacy_tag"),
    refreshTime = cms.uint64(1)

    ),
cms.PSet(
    record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
    tag = cms.string('BSOnLineHLT_tag'),
    refreshTime = cms.uint64(1)
    )

),
                                        connect = cms.string('sqlite_file:BeamSpotOnlineLegacy_JetHT.db')
                                        #connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
                                        #connect = cms.string('frontier://PromptProd/CMS_COND_31X_BEAMSPOT')
                                        )

"""
from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
process.GlobalTag = customiseGlobalTag(globaltag = "auto:run3_hlt_GRun")
#process.BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")
process.BeamSpotESProducer = cms.ESProducer("OfflineToTransientBeamSpotESProducer")

process.source = cms.Source("EmptySource")
process.source.numberEventsInRun=cms.untracked.uint32(1)
process.source.firstRun = cms.untracked.uint32(123741)
process.source.firstLuminosityBlock = cms.untracked.uint32(1)
process.source.numberEventsInLuminosityBlock = cms.untracked.uint32(2)
process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(7)
                    )
process.beamspot = cms.EDAnalyzer("OnlineBeamSpotFromDB")


process.p = cms.Path(process.beamspot)

