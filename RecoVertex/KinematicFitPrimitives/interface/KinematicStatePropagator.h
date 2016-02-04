#ifndef KinematicStatePropagator_H
#define KinematicStatePropagator_H

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

/**
 * Pure abstract base class to create
 * KinematicStatePropagators
 *
 * Kirill Prokofiev, March 2003
 */


class KinematicStatePropagator
{
public:

 KinematicStatePropagator()
 {}
 
 virtual ~KinematicStatePropagator(){}
 
/**
 * Method propagating the  KinematicState to the point of
 * closest approach at the transverse plane
 */ 
 
 virtual KinematicState propagateToTheTransversePCA(const KinematicState& state, const GlobalPoint& point) const = 0;

/**
 * Clone method
 */
 virtual KinematicStatePropagator * clone() const = 0;

private:


};
#endif
