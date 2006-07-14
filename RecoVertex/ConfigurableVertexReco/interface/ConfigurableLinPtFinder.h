#ifndef _ConfigurableLinPtFinder_H_
#define _ConfigurableLinPtFinder_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableLinPtFinder : public AbstractConfReconstructor
{
  public:
    ConfigurableLinPtFinder ();
    void configure ( const edm::ParameterSet & );
    ConfigurableLinPtFinder ( const ConfigurableLinPtFinder & o );
    ~ConfigurableLinPtFinder();
    ConfigurableLinPtFinder * clone () const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const;
    edm::ParameterSet defaults() const;
  private:
    const VertexReconstructor * theRector;
};

#endif
