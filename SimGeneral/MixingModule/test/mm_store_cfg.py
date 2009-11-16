import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODMIXNEW")
process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/F4AC0278-96BD-DE11-8687-00261894392D.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/E2E078D8-C0BD-DE11-A77B-0026189438E0.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/E0D7EB2D-90BD-DE11-9BB6-0017312B5651.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/C6D851A8-8CBD-DE11-8714-001A92971B5E.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/BCA3D46D-8DBD-DE11-AEC8-001BFCDBD15E.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/B6E6151C-8EBD-DE11-97C4-001731AF68CF.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/AE7008D9-93BD-DE11-8640-001A928116F2.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/A46DD10C-95BD-DE11-948B-00304867905A.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/8871A115-8EBD-DE11-98A9-001731AF68B9.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/86ADB9EB-0FBE-DE11-953B-001A92971AA8.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/7E39E697-9ABD-DE11-8683-001A928116FA.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/680CAFF8-95BD-DE11-A3E0-001A92971B3A.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/521815A3-94BD-DE11-9896-0018F3D096CA.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/504727E7-8EBD-DE11-93EB-003048678D52.root',
       '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/2CAB4431-97BD-DE11-8B1E-002618943972.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_*_*_PRODMIXNEW',
	'keep PSimHits*_*_*_*',
        'keep PCaloHits*_*_*_*',
        'keep SimTracks*_*_*_*',
        'keep SimVertexs*_*_*_*',
        'keep edmHepMCProduct_*_*_*'
),
    fileName = cms.untracked.string('file:/tmp/ebecheva/Cum_playbackExtended_store_TTBar5evs.root')

)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('mix'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MixingModule = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        )
    ),
    categories = cms.untracked.vstring('MixingModule'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.mix)
process.outpath = cms.EndPath(process.out)

process.mix.input.type = 'fixed'
process.mix.input.nbPileupEvents = cms.PSet(
    averageNumber = cms.double(3.0)
)

