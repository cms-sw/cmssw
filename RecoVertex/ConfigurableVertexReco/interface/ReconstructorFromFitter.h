#ifndef _ReconstructorFromFitter_H_
#define _ReconstructorFromFitter_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ReconstructorFromFitter : public VertexReconstructor
{
  public:
    ReconstructorFromFitter ( const VertexFitter &, int verbose=0 );
    ReconstructorFromFitter ( const ReconstructorFromFitter & o );
    ~ReconstructorFromFitter();
    std::vector < TransientVertex > vertices ( const std::vector < reco::TransientTrack > & ) const;

    ReconstructorFromFitter * clone () const;

  private:
    const VertexFitter * theFitter;
    int theVerbosity;
};

#endif
