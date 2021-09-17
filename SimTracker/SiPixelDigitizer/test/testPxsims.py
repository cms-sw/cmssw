
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("simTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('PixelSimHitsTest')
)

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(
#    '/store/user/kotlinski/mu100/simhits/simHits.root',
#    'file:simHits.root'
    'file:/afs/cern.ch/work/d/dkotlins/public//MC/mu/pt100/simhits/simHits1.root'
    )
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('sim_histos.root')
)

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
#process.GlobalTag.globaltag = 'MC_53_V15::All'
#process.GlobalTag.globaltag = 'DESIGN53_V15::All'
#process.GlobalTag.globaltag = 'START53_V15::All'
# ideal
process.GlobalTag.globaltag = 'MC_70_V1::All'
# realistiv alignment and calibrations 
#process.GlobalTag.globaltag = 'START70_V1::All'

process.analysis =  cms.EDAnalyzer("PixelSimHitsTest",
	src = cms.string("g4SimHits"),
#	list = cms.string("TrackerHitsPixelBarrelLowTof"),
#	list = cms.string("TrackerHitsPixelBarrelHighTof"),
	list = cms.string("TrackerHitsPixelEndcapLowTof"),
#	list = cms.string("TrackerHitsPixelEndcapHighTof"),
        Verbosity = cms.untracked.bool(False),
#        mode = cms.untracked.string("bpix"),
        mode = cms.untracked.string("fpix"),
)

process.p = cms.Path(process.analysis)

