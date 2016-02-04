#ifndef MonteCarloLinearizationPointFinder_H
#define MonteCarloLinearizationPointFinder_H

#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

  /** 
   *  A linearization point finder that can be fed with the result.
   *  Naturally, this is for debugging only.
   */

class MonteCarloLinearizationPointFinder : public LinearizationPointFinder
{
public:
  MonteCarloLinearizationPointFinder();

  /**
   *  The method that allows cheating
   */
  void setPoint ( const GlobalPoint & pos );
    
  virtual GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> & ) const;
  virtual GlobalPoint getLinearizationPoint(const std::vector<FreeTrajectoryState> & ) const;

  virtual MonteCarloLinearizationPointFinder * clone() const
  {
    return new MonteCarloLinearizationPointFinder ( * this );
  };
private:
  GlobalPoint thePt;
};

#endif
