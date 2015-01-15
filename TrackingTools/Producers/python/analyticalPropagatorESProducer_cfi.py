import FWCore.ParameterSet.Config as cms

anyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "anyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)

alongMomentumAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "alongMomentumAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)

oppositeToMomentumAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "oppositeToMomentumAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)

