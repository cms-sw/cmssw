#ifndef _AdaptiveVertexReconstructor_H_
#define _AdaptiveVertexReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <set>

class AdaptiveVertexReconstructor : public VertexReconstructor {
public:

  /***
   *
   * \paramname primcut sigma_cut for the first iteration
   *   (primary vertex)
   * \paramname seccut sigma_cut for all subsequent vertex fits.
   * \paramname minweight the minimum weight for a track to
   * stay in a fitted vertex
   */
  AdaptiveVertexReconstructor( float primcut = 2.0, float seccut = 6.0,
                               float minweight = 0.5 );

  /**
   *  The ParameterSet should have the following defined:
   *  double primcut
   *  double seccut
   *  double minweight
   *  for descriptions see 
   */
  AdaptiveVertexReconstructor( const edm::ParameterSet & s );

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> & v ) const;
  
  std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> &, const reco::BeamSpot & ) const; 

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
   *  the actual fit to avoid code duplication
   */
  std::vector<TransientVertex> 
    vertices( const std::vector<reco::TransientTrack> &, const reco::BeamSpot &,
              bool usespot ) const; 

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
