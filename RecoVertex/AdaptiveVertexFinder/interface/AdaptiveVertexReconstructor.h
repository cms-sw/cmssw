#ifndef _AdaptiveVertexReconstructor_H_
#define _AdaptiveVertexReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include <set>

class AdaptiveVertexReconstructor : public VertexReconstructor {
public:

  /***
   * 
   * \paramname primcut sigma_cut for the first iteration
   *   (primary vertex)
   * \paramname seccut sigma_cut for all subsequent vertex fits.
   * \paramname min_weight the minimum weight for a track to 
   * stay in a fitted vertex
   */
  AdaptiveVertexReconstructor( float primcut = 3.0, float seccut = 15.0, 
                               float min_weight = 0.5 );

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> & v ) const;

  virtual AdaptiveVertexReconstructor * clone() const {
    return new AdaptiveVertexReconstructor( * this );
  }

  /**
   *  tracks with a weight < 10^-8 are moved from vertex
   *  to remainingtrks container.
   */
  TransientVertex cleanUp ( const TransientVertex & old ) const;

private:

  /**
   *  contrary to what its name has you believe, ::erase removes all
   *  newvtx.originalTracks() above theMinWeight from remainingtrks.
   */
  void erase ( const TransientVertex & newvtx,
             std::set < reco::TransientTrack > & remainingtrks ) const;

private:
  float thePrimCut;
  float theSecCut;
  float theMinWeight;
};

#endif
