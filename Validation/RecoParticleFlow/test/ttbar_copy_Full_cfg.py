import FWCore.ParameterSet.Config as cms
 

process = cms.Process("COPY")
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_1.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_2.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_3.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_4.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_5.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_6.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_7.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_8.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_9.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_10.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_11.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_12.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_13.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_14.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_15.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_16.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_17.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_18.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_19.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_20.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_21.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_22.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_23.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_24.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_25.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_26.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_27.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_28.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_29.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_30.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_31.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_32.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_33.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_34.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_35.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_36.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_37.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_38.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_39.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_40.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_41.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_42.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_43.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_44.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_45.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_46.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_47.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_48.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_49.root',
     'rfio:/castor/cern.ch/user/p/pjanot/CMSSW220pre1/aod_ttbar_Full_50.root'
     )
)

process.display = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('aod_ttbar_Full.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.display)

