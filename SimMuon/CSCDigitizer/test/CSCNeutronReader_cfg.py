import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDigitizerTest")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load('Configuration.StandardSequences.Services_cff')

process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")
print str(process.RandomNumberGeneratorService)
process.mix.input.fileNames = cms.untracked.vstring(['file:cscNeutronWriter.root'])
#process.mix.mixObjects.mixSH.input.append(cms.InputTag('cscNeutronWriter'))
process.mix.mixObjects.mixSH.input = cms.VInputTag(cms.InputTag('cscNeutronWriter'))
process.mix.mixObjects.mixSH.subdets = cms.vstring('MuonCSCDigis')
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_31X::All"
process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")

process.source = cms.Source("EmptySource")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        simMuonCSCDigis = cms.untracked.uint32(468),
        mix = cms.untracked.uint32(1234)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.DQMStore = cms.Service("DQMStore")
process.simMuonCSCDigis.InputCollection = 'cscNeutronWriter'
process.dump = cms.EDFilter("CSCDigiDump",
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    empt = cms.InputTag(""),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        MixingModule = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('MixingModule'),
    destinations = cms.untracked.vstring('cout')
)

#process.o1 = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('cscdigis.root')
#)

process.p1 = cms.Path(process.mix*process.simMuonCSCDigis*process.dump)
#process.ep = cms.EndPath(process.o1)
#

