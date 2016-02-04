#ifndef FallbackLinearizationPointFinder_H
#define FallbackLinearizationPointFinder_H

#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/HsmModeFinder3d.h"

  /** 
   *  A fallback linearization point finder that is
   *  used if the 'actual' CrossingPtBasedLinPtFinder fails.
   *  Computes the mode based on innermost states.
   */

class FallbackLinearizationPointFinder : public LinearizationPointFinder
{
public:
  FallbackLinearizationPointFinder ( const ModeFinder3d & m = HsmModeFinder3d() );
  virtual GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> & ) const;
  virtual GlobalPoint getLinearizationPoint(const std::vector<FreeTrajectoryState> & ) const;

  virtual FallbackLinearizationPointFinder * clone() const
  {
    return new FallbackLinearizationPointFinder ( * this );
  };
private:
  ModeFinder3d * theModeFinder;
};

#endif
