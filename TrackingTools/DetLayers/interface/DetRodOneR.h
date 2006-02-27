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
  typedef vector<GeometricSearchDet*> DetContainer;

  /// Dummy constructor
  DetRodOneR(){};

  /// Construct from iterators on Det*
  DetRodOneR( vector<const GeomDet*>::const_iterator first,
	      vector<const GeomDet*>::const_iterator last);

  /// Construct from a vector of Det*
  DetRodOneR( const vector<const GeomDet*>& dets);

  virtual ~DetRodOneR();

  virtual vector<const GeomDet*> basicComponents() const {return theDets;}


protected:
  /// Query detector idet for compatible and add the output to result.
  
  bool add( int idet, vector<DetWithState>& result,
	    const TrajectoryStateOnSurface& startingState,
	    const Propagator& prop, 
	    const MeasurementEstimator& est) const;

 private:
  vector<const GeomDet*>     theDets;
  
  void initialize();

};

#endif
