#ifndef _ConfigurableKalmanFitter_H_
#define _ConfigurableKalmanFitter_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableKalmanFitter : public AbstractConfReconstructor
{
  public:
    ConfigurableKalmanFitter ();
    void configure ( const edm::ParameterSet & );
    ConfigurableKalmanFitter ( const ConfigurableKalmanFitter & o );
    ~ConfigurableKalmanFitter();
    ConfigurableKalmanFitter * clone () const;
    edm::ParameterSet defaults() const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const;
  private:
    const VertexReconstructor * theRector;
};

#endif
