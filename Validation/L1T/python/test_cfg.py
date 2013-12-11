import FWCore.ParameterSet.Config as cms

process = cms.Process("L1Val")

process.load("Validation.L1T.L1Validator_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
#        'file:step2_RAW2DIGI_RECO_VALIDATION_DQM.root'
	'/store/relval/CMSSW_6_2_0_pre8/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/3AFDBE70-E2E0-E211-B9A5-003048F179C2.root'
#	'/store/relval/CMSSW_7_0_0_pre1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/A8B55F84-2E0F-E311-8F27-003048D15E24.root'
    )
)

process.p = cms.Path(process.L1Validator)
