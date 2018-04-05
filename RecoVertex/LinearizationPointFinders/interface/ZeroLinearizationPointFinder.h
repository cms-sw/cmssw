#ifndef ZeroLinearizationPointFinder_H
#define ZeroLinearizationPointFinder_H

#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"

  /** 
   *  A linearization point finder that always returns (0,0,0)
   */

class ZeroLinearizationPointFinder : public LinearizationPointFinder
{
public:
  GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> & ) const override;
  GlobalPoint getLinearizationPoint(const std::vector<FreeTrajectoryState> & ) const override;

  ZeroLinearizationPointFinder * clone() const override
  {
    return new ZeroLinearizationPointFinder ( * this );
  };
};

#endif
