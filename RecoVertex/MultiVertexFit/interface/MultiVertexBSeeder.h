#ifndef _MultiVertexBSeeder_H_
#define _MultiVertexBSeeder_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

/**
 *  A good seeder for "B-jetty" setups
 *  (i.e. high-multiplicity, collimated track "bundles" with
 *  at least one secondary vertex )
 */
class MultiVertexBSeeder : public VertexReconstructor
{
public:
  MultiVertexBSeeder ( double nsigma=50. );
  std::vector<TransientVertex> vertices(
      const std::vector<reco::TransientTrack> &) const; 

  MultiVertexBSeeder * clone() const;

private:
  double theNSigma;

};

#endif
