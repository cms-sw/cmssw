#ifndef DetLayers_ForwardDetRingOneZ_H
#define DetLayers_ForwardDetRingOneZ_H

/** \class ForwardDetRingOneZ
 *  A ring of detectors, all having the same BoundDisk.
 */

#include "TrackingTools/DetLayers/interface/ForwardDetRing.h"


class ForwardDetRingOneZ : public ForwardDetRing {
public:

  /// Construct from iterators on Det*.
  ForwardDetRingOneZ( vector<const GeomDet*>::const_iterator first,
		      vector<const GeomDet*>::const_iterator last);

  // Construct from a vector of Det*.
  ForwardDetRingOneZ( const vector<const GeomDet*>& dets);

  virtual ~ForwardDetRingOneZ();
  
  virtual vector<const GeomDet*> basicComponents() const {return theDets;}

protected:

  /// Query detector idet for compatible and add the output to result.
  /*
  bool add( int idet, vector<DetWithState>& result,
 	    const FreeTrajectoryState& fts,
 	    const Propagator& prop, 
 	    const MeasurementEstimator& est) const;
  */

private:
  vector<const GeomDet*>     theDets;

  void initialize();

};
#endif

