import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOEleSkim")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( '/store/relval/CMSSW_3_2_6/RelValZEE/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/9AB61A32-BA9A-DE11-8298-001D09F29619.root',
                                                               '/store/relval/CMSSW_3_2_6/RelValZEE/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/80CF3175-C59A-DE11-AB6E-0016177CA778.root',
                                                               '/store/relval/CMSSW_3_2_6/RelValZEE/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/7047D992-FC9A-DE11-863B-0019B9F72BAA.root',
                                                               '/store/relval/CMSSW_3_2_6/RelValZEE/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/5CC96C97-B59A-DE11-8CAA-0019B9F6C674.root',
                                                               '/store/relval/CMSSW_3_2_6/RelValZEE/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/126AA31F-BE9A-DE11-97E9-001D09F290BF.root'
                                                               )
                            )


#load the EventContent and Skim cff/i files for the sub-skim.
process.load('SUSYBSMAnalysis.Skimming.EXOEle_EventContent_cfi')
process.load('SUSYBSMAnalysis.Skimming.EXOEle_cff')

#define output file name.
process.exoticaEleOutputModule.fileName = cms.untracked.string('EXOEle.root')

#all three paths need to run so that the Oputput module can keep the logcal "OR"
process.exoticaEleLowetPath =cms.Path(process.exoticaEleLowetSeq)
process.exoticaEleMedetPath =cms.Path(process.exoticaEleMedetSeq)
process.exoticaEleHighetPath=cms.Path(process.exoticaEleHighetSeq)


process.endPath = cms.EndPath(process.exoticaEleOutputModule)
