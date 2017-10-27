#ifndef _AdaptiveVertexReconstructor_H_
#define _AdaptiveVertexReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
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
   * \paramname smoothing perform track smoothing?
   */
  AdaptiveVertexReconstructor( float primcut = 2.0, float seccut = 6.0,
                               float minweight = 0.5, bool smoothing = false );

  ~AdaptiveVertexReconstructor() override;

  /**
   *  The ParameterSet should have the following defined:
   *  double primcut
   *  double seccut
   *  double minweight
   *  for descriptions see 
   */
  AdaptiveVertexReconstructor( const edm::ParameterSet & s );

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> & v ) const override;
  
  std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> &, const reco::BeamSpot & ) const override; 
  
  std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> & primaries,
             const std::vector<reco::TransientTrack> & tracks, 
             const reco::BeamSpot & ) const override; 

  AdaptiveVertexReconstructor * clone() const override {
    return new AdaptiveVertexReconstructor( * this );
  }

private:
  /**
   *  the actual fit to avoid code duplication
   */
  std::vector<TransientVertex> 
    vertices( const std::vector<reco::TransientTrack> & primaries,
              const std::vector<reco::TransientTrack> & trks,
              const reco::BeamSpot &, bool has_primaries, bool usespot ) const; 

  /**
   *  contrary to what its name has you believe, ::erase removes all
   *  newvtx.originalTracks() above theMinWeight from remainingtrks.
   */
  void erase ( const TransientVertex & newvtx,
             std::set < reco::TransientTrack > & remainingtrks, float w ) const;

  /**
   *  cleanup reconstructed vertices. discard all with too few significant
   *  tracks.
   */
  std::vector<TransientVertex> cleanUpVertices ( 
      const std::vector < TransientVertex > & ) const;

  /** setup the vertex fitters.
   */
  void setupFitters ( float primcut, float primT, float primr,
      float seccut, float secT, float secr, bool smoothing );

  TransientVertex cleanUp ( const TransientVertex & old ) const;

private:
  AdaptiveVertexFitter * thePrimaryFitter; // one fitter for the primaries ...
  AdaptiveVertexFitter * theSecondaryFitter; // ... and one for the rest.

  // the minimum weight for a track to stay in a vertex.
  float theMinWeight;
  // the minimum weight for a track to be considered "significant".
  float theWeightThreshold;
};

#endif
