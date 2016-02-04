#ifndef _LinearizationPointFinder_H_
#define _LinearizationPointFinder_H_

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>
class FreeTrajectoryState;

/**  Generic class to make an Initial Linearization point
 */

class LinearizationPointFinder{

public:
  virtual ~LinearizationPointFinder() {}

  /** Virtual method returning the Initial Linearization Point
   *  as an object of type GlobalPoint
   */

  virtual GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> &)
    const=0;

  virtual GlobalPoint getLinearizationPoint(const std::vector<FreeTrajectoryState> &) const;

  /**
   *  Clone method
   */
   virtual LinearizationPointFinder * clone() const=0;

};

#endif
