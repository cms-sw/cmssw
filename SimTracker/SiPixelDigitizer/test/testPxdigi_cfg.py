#
import FWCore.ParameterSet.Config as cms

process = cms.Process("digiTest")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
# process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('PixelDigisTest'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(
#    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/digis/digis2_postls171.root'
    'file:digis.root'
    )
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

#process.GlobalTag.globaltag = 'MC_31X_V9::All'
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'
# 2012
# process.GlobalTag.globaltag = 'GR_P_V40::All'
# 2013 MC
#process.GlobalTag.globaltag = 'MC_70_V1::All'
# 2014 data
process.GlobalTag.globaltag = 'PRE_R_71_V3::All'
# 2014 mc
#process.GlobalTag.globaltag = 'PRE_STA71_V4::All'

  
process.analysis = cms.EDAnalyzer("PixelDigisTest",
    Verbosity = cms.untracked.bool(True),
# my sim in V7
#    src = cms.InputTag("simSiPixelDigis"),
#    src = cms.InputTag("mix"),
# old default
    src = cms.InputTag("siPixelDigis"),
)

process.p = cms.Path(process.analysis)

