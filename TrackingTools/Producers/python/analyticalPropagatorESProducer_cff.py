import FWCore.ParameterSet.Config as cms

from TrackingTools.GeomPropagators.AnalyticalPropagator_cfi import AnalyticalPropagator

anyDirectionAnalyticalPropagator = AnalyticalPropagator.clone(
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "anyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)

alongMomentumAnalyticalPropagator = AnalyticalPropagator.clone(
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "alongMomentumAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)

oppositeToMomentumAnalyticalPropagator = AnalyticalPropagator.clone(
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "oppositeToMomentumAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)

