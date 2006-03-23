#ifndef ZeroLinearizationPointFinder_H
#define ZeroLinearizationPointFinder_H

#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"

  /** 
   *  A linearization point finder that always returns (0,0,0)
   */

class ZeroLinearizationPointFinder : public LinearizationPointFinder
{
public:
  virtual GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> & ) const;
  virtual GlobalPoint getLinearizationPoint(const std::vector<FreeTrajectoryState> & ) const;

  virtual ZeroLinearizationPointFinder * clone() const
  {
    return new ZeroLinearizationPointFinder ( * this );
  };
};

#endif
