#ifndef _ReconstructorFromLinPtFinder_H_
#define _ReconstructorFromLinPtFinder_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"

/**
 *  Wrap any LinearizationPointFinder into the VertexReconstructor interface
 */

class ReconstructorFromLinPtFinder : public VertexReconstructor
{
  public:
    ReconstructorFromLinPtFinder ( const LinearizationPointFinder &, int verbose=0 );
    ReconstructorFromLinPtFinder ( const ReconstructorFromLinPtFinder & o );
    ~ReconstructorFromLinPtFinder();
    std::vector < TransientVertex > vertices ( const std::vector < reco::TransientTrack > & ) const;

    ReconstructorFromLinPtFinder * clone () const;

  private:
    const LinearizationPointFinder * theLinPtFinder;
    int theVerbosity;
};

#endif
