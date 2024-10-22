import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("SimTauAnalyzer")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = "DEBUG"
process.MessageLogger.debugModules = ["SimTauCPLink", "SimTauProducer", "SimTauAnalyzer"]

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1)
                                        )
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
                                'file:/data/agruber/patatrack/CMSSW_14_0_0_pre0/src/24036.0_ZTT_14TeV+2026D96/step3.root'
                                #'file:SimTauProducer_test.root'
                                )
                            )

process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string('test.root')
                                   )

process.SimTauAnalyzer = cms.EDAnalyzer('SimTauAnalyzer',
simTau            = cms.InputTag('SimTauCPLink')
                              )

process.p = cms.Path(process.SimTauAnalyzer)
