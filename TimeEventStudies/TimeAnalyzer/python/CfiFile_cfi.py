import FWCore.ParameterSet.Config as cms

TimeAnalysis = cms.EDAnalyzer('TimeAnalyzer',
                              acceptedParticleTypes = cms.vint32( 11,-11 )
                              )
