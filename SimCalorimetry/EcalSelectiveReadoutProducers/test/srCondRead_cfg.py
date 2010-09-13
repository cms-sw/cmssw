import FWCore.ParameterSet.Config as cms

process = cms.Process("P")

#Conditions:
#process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_38Y_V10::All'

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
)

process.ecalSRCondTools = cms.EDAnalyzer("EcalSRCondTools",
                                         mode = cms.string("read")
)


## ----------------------------------------------------------------------
## To read SR configuration from an sqlite file:
##
## #tag = 'EcalSRSettings_v00_lowlumi_mc'
## tag = 'EcalSRSettings_v00_beam10_mc'
## 
## process.GlobalTag.toGet = cms.VPSet(
##       cms.PSet(record = cms.string("EcalSRSettingsRcd"),
##                tag = cms.string(tag),
##                connect = cms.untracked.string('sqlite_file:' + tag + '.db')
##       )
## )
## ----------------------------------------------------------------------

process.p1 = cms.Path(process.ecalSRCondTools)
