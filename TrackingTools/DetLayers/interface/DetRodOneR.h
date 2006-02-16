#ifndef DetLayers_DetRodOneR_H
#define DetLayers_DetRodOneR_H

/** \class DetRodOneR
 *  A rod of detectors, all having the same BoundPlane.
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class MeasurementEstimator;

class DetRodOneR : public DetRod {
 public: 
  typedef GeometricSearchDet Det;
  typedef GSDUnit DetUnit;
  typedef vector<GeometricSearchDet*> DetContainer;

  /// Construct from iterators on Det*.
  DetRodOneR( vector<Det*>::const_iterator first,
	      vector<Det*>::const_iterator last);

  /// Construct from a vector of Det*
  DetRodOneR( const vector<Det*>& dets);

  /// Construct from a vector of DetUnit*
  DetRodOneR( const vector<GSDUnit*>& detUnits);

  virtual ~DetRodOneR();

  virtual const DetContainer& dets() const {return theDets;}

protected:
  
  /// Query detector idet for compatible and add the output to result.
  bool add( int idet, vector<DetWithState>& result,
	    const TrajectoryStateOnSurface& startingState,
	    const Propagator& prop, 
	    const MeasurementEstimator& est) const;

 private:
  DetContainer     theDets;

  void initialize();

};

#endif
