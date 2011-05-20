import FWCore.ParameterSet.Config as cms

process = cms.Process("Summary")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_42_V14::All" 



process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
    #'file:/uscms/home/aeverett/work/CMSSW_4_2_3/src/skimSingleMuonAll.root'
    #'file:skimAll.root'
    #'file:DoubleMu/res/merge.root'
    'file:DoubleElectron/res/skimAll_10_1_2JY.root',
    'file:DoubleElectron/res/skimAll_1_1_wv1.root',
    'file:DoubleElectron/res/skimAll_2_1_FKd.root',
    'file:DoubleElectron/res/skimAll_4_1_5Ox.root',
    'file:DoubleElectron/res/skimAll_5_1_jdQ.root',
    'file:DoubleElectron/res/skimAll_6_1_IDK.root',
    'file:DoubleElectron/res/skimAll_8_1_utp.root',
    'file:DoubleElectron/res/skimAll_9_1_Ueg.root',
    
    )
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("histoDE.root") )
process.load("SUSYBSMAnalysis.Skimming.skimSummary_cfi")
#process.HotLineSummary.histoFileName = cms.untracked.string('hotLineSummaryPlot.root')
process.SkimSummary.HltLabel = cms.InputTag("TriggerResults","","SKIM")
#process.SkimSummary.HltLabel = cms.InputTag("TriggerResults","","HLT")

#JEC
#process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

process.p = cms.Path(process.SkimSummary)

process.MessageLogger.categories.append('HLTrigReport')

#process.load( "HLTrigger.HLTanalyzers.hlTrigReport_cfi" )
#process.hlTrigReport.HLTriggerResults   = cms.InputTag("TriggerResults", "", "SKIM")
#process.hlTrigReport.ReferencePath      = cms.untracked.string("Skim_diMuons") #( "HLTriggerFinalPath" )
#process.hlTrigReport.ReferenceRate      = cms.untracked.double( 100.0 )
#process.report = cms.EndPath( process.hlTrigReport )
