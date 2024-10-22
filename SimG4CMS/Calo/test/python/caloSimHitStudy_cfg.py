import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process("Analysis",Run3_DDD)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")
process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HitStudy=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#                               'file:step1_run3Old.root'
                                'file:step1_run3New.root',
                            )
                        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(
#                                      'analRun3Old.root'
                                       'analRun3New.root'
                                   )
)

process.analysis_step   = cms.Path(process.CaloSimHitStudy)


# Schedule definition                                                          
process.schedule = cms.Schedule(process.analysis_step)
