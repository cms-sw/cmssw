import FWCore.ParameterSet.Config as cms
 

process = cms.Process("COPY")
 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==0.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==1.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==2.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==3.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==4.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==5.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==6.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==7.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==8.root',
     'rfio:==CASTOR==/==TYPE==_==NAME==_==SIMU==_==JOBIN==9.root'
     )
)

process.display = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('==TYPE==_==NAME==_==SIMU==_00==JOB==.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)

process.outpath = cms.EndPath(process.display)

