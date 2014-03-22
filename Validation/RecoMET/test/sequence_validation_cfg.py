import FWCore.ParameterSet.Config as cms

process = cms.Process("METVALIDATION")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

#process.GlobalTag.globaltag = 'START42_V17::All'
##process.GlobalTag.globaltag = 'MC_38Y_V14::All'
## for 6_2_0 QCD
process.GlobalTag.globaltag = 'PRE_ST62_V8::All'

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#
#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")


process.load("Validation.RecoMET.METRelValForDQM_cff")


process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    #debugFlag = cms.untracked.bool(True),
    #debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_6_2_0/RelValTTbarLepton/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/44754556-57EC-E211-A3EE-003048FEB9EE.root',
#    '/store/relval/CMSSW_6_2_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/PRE_ST62_V8-v3/00000/604E213E-49EC-E211-9D8D-003048F0E5CE.root'        
    )


)

#process.Tracer = cms.Service("Tracer")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


#process.fileSaver = cms.EDAnalyzer("METFileSaver",
#                                 OutputFile = cms.untracked.string('METTesters.root')
#) 
process.p = cms.Path(#process.fileSaver*
                     #                     process.genMetTrue*
                     #                     process.genMetCalo*
                     #                     process.genMetCaloAndNonPrompt*
                     #                     process.tcMet*
                     process.METValidation
)
process.DQM.collectorHost = ''

#process.dqmoffline_step = cms.Path(process.dqmStoreStats)

#process.schedule = cms.Schedule(process.p,process.dqmoffline_step)


