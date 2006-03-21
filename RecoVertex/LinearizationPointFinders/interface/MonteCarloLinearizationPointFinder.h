#ifndef MonteCarloLinearizationPointFinder_H
#define MonteCarloLinearizationPointFinder_H

#include "RecoRecoVertex/VertexTools/interface/LinearizationPointFinder.h"

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
    
  virtual GlobalPoint getLinearizationPoint(const vector<DummyRecTrack> & ) const;
  virtual GlobalPoint getLinearizationPoint(const vector<FreeTrajectoryState> & ) const;

  virtual MonteCarloLinearizationPointFinder * clone() const
  {
    return new MonteCarloLinearizationPointFinder ( * this );
  };
private:
  GlobalPoint thePt;
};

#endif
