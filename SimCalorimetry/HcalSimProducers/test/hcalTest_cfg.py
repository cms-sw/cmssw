# New Mixer test #1 - based on:
#
# Auto generated configuration file
# using: 
# Revision: 1.125 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: SinglePiPt100.cfi -s GEN:ProductionFilterSequence,SIM,DIGI,DATAMIX,L1,DIGI2RAW,HLT -n 10 --conditions FrontierConditions_GlobalTag,STARTUP31X_V1::All --eventcontent FEVTDEBUG --no_exec
#
# Hacked to accept exising MC files as input
#
import FWCore.ParameterSet.Config as cms

process = cms.Process('MIXTEST3')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/Digi_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('SinglePiPt100.cfi nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# Input source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)
process.source = cms.Source( 
    "PoolSource"
,fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre4/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V12-v1/0001/D0B6FD82-12C9-DE11-9245-00304879FBB2.root')
)



# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'MC_3XY_V12::All'
process.simHcalUnsuppressedDigis.ho.siPMCode = 0

#process.load("CalibCalorimetry/HcalPlugins/Hcal_FrontierConditions_cff")
##process.es_pool.toGet[2].tag = 'HcalGains_v4.04_mc'
#from CondCore.DBCommon.CondDBSetup_cfi import *

#process.es_pool = cms.ESSource("PoolDBESSource",
#    CondDBSetup,
#   connect = cms.string('frontier://FrontierProd/CMS_COND_31X_HCAL'), ##FrontierDev/CMS_COND_HCAL"
#    toGet = cms.VPSet(
#        cms.PSet(
#           record = cms.string('HcalGainsRcd'),
#           tag = cms.string('HcalGains_v4.04_mc')
#       )
#   )
#)
#process.es_prefer_es_pool = cms.ESPrefer("PoolDBESSource","es_pool")


#process.generation_step = cms.Path(process.pgen)
#process.simulation_step = cms.Path(process.psim)
process.dump = cms.EDAnalyzer("HcalDigiDump")
#process.load("RecoLocalCalo/HcalRecProducers/HcalHitReconstructor_ho_cfi")
process.load("RecoLocalCalo/Configuration/hcalLocalReco_cff")
process.horeco.digiLabel = 'simHcalUnsuppressedDigis'
process.hbhereco.digiLabel = 'simHcalUnsuppressedDigis'
process.hfreco.digiLabel = 'simHcalUnsuppressedDigis'

process.eventContent = cms.EDAnalyzer("EventContentAnalyzer")
process.load("Validation/HcalRecHits/HcalRecHitParam_cfi")
process.hcalRecoAnalyzer.eventype = 'multi'
process.hcalRecoAnalyzer.outputFile = 'ho.root'
process.hcalRecoAnalyzer.hcalselector = 'HO'
process.hcalRecoAnalyzer.ecalselector = 'no'
process.path = cms.Path(process.mix+process.simHcalUnsuppressedDigis+process.horeco+process.hcalRecoAnalyzer)
