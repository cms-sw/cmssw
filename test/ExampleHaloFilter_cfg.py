import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

isData = False
process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.load("RecoMET/Configuration/RecoMET_BeamHaloId_cff")
process.load('RecoMET.METAnalyzers.CSCHaloFilter_cfi')

# Expected BX for ALCT Digis is (FOR MC = 6, FOR DATA =3)
if isData:
    process.GlobalTag.globaltag = 'GR10_P_V12::All'
else:
    process.GlobalTag.globaltag = 'START39_V8::All'


process.load("RecoMuon/Configuration/RecoMuon_cff")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(

     '/store/relval/CMSSW_3_9_7/RelValBeamHalo/GEN-SIM-RECO/START39_V8-v1/0047/08704CC8-830D-E011-9AD3-002618943852.root'
     
    
    ),
)



process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.p = cms.Path(process.CSCTightHaloFilter)

process.schedule = cms.Schedule(
    process.p
    )



                                         
                                     
