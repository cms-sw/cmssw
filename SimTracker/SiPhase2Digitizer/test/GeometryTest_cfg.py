import FWCore.ParameterSet.Config as cms

process = cms.Process("digiTest")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelRawData'),
    destinations = cms.untracked.vstring("cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
)
process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(
       'file:/afs/cern.ch/user/d/dutta/work/public/Digitizer/CMSSW_6_2_SLHCDEV_X_2015-07-19-1100/src/10000_FourMuPt1_200/step1.root'
       )
)
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
#process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2017_cff')
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2017', '')
#-------------
# Output ROOT file
#-------------
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('./GeometryTest.root')
)
process.analysis = cms.EDAnalyzer("GeometryTest",
    hitsProducer = cms.string('g4SimHits'),
    ROUList = cms.vstring(
        'TrackerHitsPixelBarrelLowTof',
        'TrackerHitsPixelEndcapLowTof',
        'TrackerHitsPixelEndcapHighTof',
        'TrackerHitsPixelBarrelHighTof',
        'TrackerHitsTECHighTof',        
        'TrackerHitsTECLowTof',
        'TrackerHitsTIBHighTof',
        'TrackerHitsTIBLowTof',
        'TrackerHitsTIDHighTof',
        'TrackerHitsTIDLowTof',
        'TrackerHitsTOBHighTof',
        'TrackerHitsTOBLowTof'),
    GeometryType = cms.string('idealForDigi')
)
process.p = cms.Path(process.analysis)
