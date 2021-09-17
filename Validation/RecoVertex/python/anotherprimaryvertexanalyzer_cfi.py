import FWCore.ParameterSet.Config as cms

primaryvertexanalyzer = cms.EDAnalyzer('AnotherPrimaryVertexAnalyzer',
                                       pvCollection = cms.InputTag("offlinePrimaryVertices"),
                                       firstOnly = cms.untracked.bool(False),
                                       vHistogramMakerPSet = cms.PSet(
                                         trueOnly = cms.untracked.bool(True),
                                         maxLSBeforeRebin = cms.uint32(100),
                                         weightThreshold = cms.untracked.double(0.5),
                                         bsConstrained = cms.bool(False),
                                         histoParameters = cms.untracked.PSet(
                                              nBinX = cms.untracked.uint32(200), xMin=cms.untracked.double(-1.), xMax=cms.untracked.double(1.),
                                              nBinY = cms.untracked.uint32(200), yMin=cms.untracked.double(-1.), yMax=cms.untracked.double(1.),
                                              nBinZ = cms.untracked.uint32(200), zMin=cms.untracked.double(-20.), zMax=cms.untracked.double(20.)
                                              )
                                       ),
                                       usePrescaleWeight = cms.bool(False),
                                       prescaleWeightProviderPSet = cms.PSet(
                                            prescaleWeightVerbosityLevel      = cms.uint32( 0 ),
                                            prescaleWeightTriggerResults      = cms.InputTag( "TriggerResults::HLT" ),
                                            prescaleWeightL1GtTriggerMenuLite = cms.InputTag( "l1GtTriggerMenuLite" ),
                                            prescaleWeightHltPaths            = cms.vstring() 
                                            ),
                                       stageL1Trigger = cms.uint32(1)
                                      )
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(primaryvertexanalyzer, stageL1Trigger = 2)
