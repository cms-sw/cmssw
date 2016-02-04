#ifndef _ReconstructorFromFitter_H_
#define _ReconstructorFromFitter_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ReconstructorFromFitter : public AbstractConfReconstructor 
{
  public:
    ReconstructorFromFitter ( const AbstractConfFitter & );
    ReconstructorFromFitter ( const ReconstructorFromFitter & o );
    ~ReconstructorFromFitter();
    void configure(const edm::ParameterSet&);
    edm::ParameterSet defaults() const;
    std::vector < TransientVertex > vertices ( const std::vector < reco::TransientTrack > & ) const;
    std::vector < TransientVertex > vertices ( const std::vector < reco::TransientTrack > &,
        const reco::BeamSpot & ) const;

    ReconstructorFromFitter * clone () const;

  private:
    const AbstractConfFitter * theFitter;
};

#endif
