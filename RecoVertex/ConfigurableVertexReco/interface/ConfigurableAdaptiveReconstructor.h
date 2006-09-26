#ifndef _ConfigurableAdaptiveReconstructor_H_
#define _ConfigurableAdaptiveReconstructor_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableAdaptiveReconstructor : public AbstractConfReconstructor
{
  public:
    ConfigurableAdaptiveReconstructor ();
    void configure ( const edm::ParameterSet & );
    ConfigurableAdaptiveReconstructor ( const ConfigurableAdaptiveReconstructor & o );
    ~ConfigurableAdaptiveReconstructor();
    ConfigurableAdaptiveReconstructor * clone () const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const;
    edm::ParameterSet defaults() const;
  private:
    const VertexReconstructor * theRector;
};

#endif
