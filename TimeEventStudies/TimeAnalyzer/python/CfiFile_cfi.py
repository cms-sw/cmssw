import FWCore.ParameterSet.Config as cms

TimeAnalysis = cms.EDAnalyzer('TimeAnalyzer',
                              acceptedParticleTypes = cms.vint32( 11,-11 ),
                              lowestenergy=cms.double( 10.0 ),
			      acceptedParentTypes = cms.vint32(23),
			      lowestenergyparent=cms.double( 70.0 ) 
                              )
