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
    void configure ( const edm::ParameterSet & ) override;
    ConfigurableAdaptiveReconstructor ( const ConfigurableAdaptiveReconstructor & o );
    ~ConfigurableAdaptiveReconstructor() override;
    ConfigurableAdaptiveReconstructor * clone () const override;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const override;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t,
        const reco::BeamSpot & ) const override;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & prims,
        const std::vector < reco::TransientTrack > & secs,
        const reco::BeamSpot & ) const override;
    edm::ParameterSet defaults() const override;
  private:
    const VertexReconstructor * theRector;
};

#endif
