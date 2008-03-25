#ifndef _MultiVertexReconstructor_H_
#define _MultiVertexReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexFitter.h"
#include "RecoVertex/MultiVertexFit/interface/DefaultMVFAnnealing.h"

/**
 *  Class that wraps the MultiVertexFitter, together with
 *  a user-supplied VertexReconstructor into a VertexReconstructor.
 */
class MultiVertexReconstructor : public VertexReconstructor
{
public:
  MultiVertexReconstructor ( const VertexReconstructor &,
                             const AnnealingSchedule & s = DefaultMVFAnnealing(),
                             float revive=-1. );
  MultiVertexReconstructor ( const MultiVertexReconstructor & );
  ~MultiVertexReconstructor();

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &) const; 
  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &,
      const std::vector < reco::TransientTrack > & primaries ) const;
  VertexReconstructor * reconstructor() const;

  MultiVertexReconstructor * clone() const;

private:
  VertexReconstructor * theOldReconstructor;
  mutable MultiVertexFitter theFitter;
};

#endif
