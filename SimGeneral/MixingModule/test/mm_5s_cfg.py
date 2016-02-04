import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODMIX")
process.load("SimGeneral.MixingModule.mixLowLumPU_5sources_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/4/5/RelVal-RelValSingleElectronPt35-1207397810/0000/20AC78F1-1403-DD11-B5CC-000423D987E0.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_*_*_PRODMIX'),
    fileName = cms.untracked.string('file:Cum_5s.root')
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
process.mix.cosmics.type = 'none'
process.mix.beamhalo_minus.type = 'none'
process.mix.beamhalo_plus.type = 'none'


