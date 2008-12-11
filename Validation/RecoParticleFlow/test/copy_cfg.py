import FWCore.ParameterSet.Config as cms
 

process = cms.Process("COPY")
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_0.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_1.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_2.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_3.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_4.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_5.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_6.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_7.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_8.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_9.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_10.root'
     )
)

process.display = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('==TYPE==_==NAME==_==SIMU==.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.display)

