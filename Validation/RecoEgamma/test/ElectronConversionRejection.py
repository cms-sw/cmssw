import FWCore.ParameterSet.Config as cms

process = cms.Process("TestElectronConversionRejectionValidator")

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



process.load("Validation.RecoEgamma.electronConversionRejectionValidator")

process.eleConvRejectionValidation.OutputFileName = 'ElectronConversionRejectionValidation.root'

process.source = cms.Source("PoolSource",
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_4_4_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/START44_V2-v1/0018/D0CE4726-80B3-E011-AC32-0030486790BA.root',

        #'/store/relval/CMSSW_4_4_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START44_V2-v1/0022/12E5D087-A6B3-E011-9A28-003048679266.root',
        #'/store/relval/CMSSW_4_4_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START44_V2-v1/0018/E04D64F4-39B2-E011-B8DD-002618943976.root',
        #'/store/relval/CMSSW_4_4_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START44_V2-v1/0018/4AC726F3-3AB2-E011-9DAD-002354EF3BCE.root',


     ),
 )


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('dqmout.root')
)


process.p1 = cms.Path(process.eleConvRejectionValidation*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
