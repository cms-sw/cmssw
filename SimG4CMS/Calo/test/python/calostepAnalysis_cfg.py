import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process("Analysis",Run2_2018)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4CMS.Calo.caloSimHitAnalysis_cfi")
process.load("Configuration.Geometry.GeometryExtended2018Reco_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('HitStudy')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                                'file:simevent_singleMuon_FTFP_BERT_EMM.root'
#       'file:simG4.root',
#       'file:simGV.root',
                            )
                        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(
                                       'analG4.root'
#                                      'analG4.root'
                                   )
)

process.analysis_step   = cms.Path(process.caloSimHitAnalysis)

#process.caloSimHitAnalysis.moduleLabel = "geantv"
#process.caloSimHitAnalysis.timeScale   = 1.0
process.caloSimHitAnalysis.passiveHits = True

# Schedule definition                                                          
process.schedule = cms.Schedule(process.analysis_step)
