#ifndef _TertiaryTracksVertexFinder_H_
#define _TertiaryTracksVertexFinder_H_

#include "RecoVertex/TertiaryTracksVertexFinder/interface/ConfigurableTertiaryTracksVertexFinder.h"

class TertiaryTracksVertexFinder : public VertexReconstructor {

  public:

  TertiaryTracksVertexFinder();

  virtual ~TertiaryTracksVertexFinder();

  virtual std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> & tracks) const {
    return theFinder->vertices(tracks); 
  }

  virtual std::vector<TransientVertex> 
    vertices(const std::vector<reco::TransientTrack> & tracks, const TransientVertex& pv) const {
    return theFinder->vertices(tracks,pv); 
  }

  virtual TertiaryTracksVertexFinder * clone() const {
    return new TertiaryTracksVertexFinder(*this);
  }

  private:

  ConfigurableTertiaryTracksVertexFinder * theFinder;

};


#endif

