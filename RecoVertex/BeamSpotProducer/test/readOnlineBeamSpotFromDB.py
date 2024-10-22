import FWCore.ParameterSet.Config as cms

process = cms.Process("readDB")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("CondCore.DBCommon.CondDBSetup_cfi")


tesOnLineESProducer = True

if tesOnLineESProducer:
    process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
                                                tag = cms.string("BeamSpotOnlineTestLegacy"),
                                                refreshTime = cms.uint64(1)

                                            ),
                                            #cms.PSet(
                                            #    record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
                                            #    tag = cms.string('BSOnLineHLT_tag'),
                                            #    refreshTime = cms.uint64(1)
                                            #    )

                                ),
                                        connect = cms.string('oracle://cms_orcon_prod/CMS_CONDITIONS')

                                        )
    process.BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")

else:
    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
    process.GlobalTag = customiseGlobalTag(globaltag = "auto:run3_hlt_GRun")
    process.BeamSpotESProducer = cms.ESProducer("OfflineToTransientBeamSpotESProducer")


process.MessageLogger.cerr.enable = False
process.MessageLogger.files.detailedInfo   = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))

process.source = cms.Source("EmptySource")
process.source.numberEventsInRun=cms.untracked.uint32(2)
process.source.firstRun = cms.untracked.uint32(336365)
process.source.firstLuminosityBlock = cms.untracked.uint32(13)
process.source.numberEventsInLuminosityBlock = cms.untracked.uint32(1)
process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(5)
                    )
process.beamspot = cms.EDAnalyzer("OnlineBeamSpotFromDB")


process.p = cms.Path(process.beamspot)

