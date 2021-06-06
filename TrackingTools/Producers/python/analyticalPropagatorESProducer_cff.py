import FWCore.ParameterSet.Config as cms

from TrackingTools.GeomPropagators.AnalyticalPropagator_cfi import AnalyticalPropagator

anyDirectionAnalyticalPropagator = AnalyticalPropagator.clone(
  MaxDPhi =  1.6 ,
  ComponentName =  "anyDirectionAnalyticalPropagator" ,
  PropagationDirection =  "anyDirection" 
)

alongMomentumAnalyticalPropagator = AnalyticalPropagator.clone(
  MaxDPhi =  1.6 ,
  ComponentName =  "alongMomentumAnalyticalPropagator" ,
  PropagationDirection =  "alongMomentum" 
)

oppositeToMomentumAnalyticalPropagator = AnalyticalPropagator.clone(
  MaxDPhi =  1.6 ,
  ComponentName =  "oppositeToMomentumAnalyticalPropagator" ,
  PropagationDirection =  "oppositeToMomentum" 
)
