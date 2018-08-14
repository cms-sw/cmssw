#ifndef _ConfigurableTrimmedKalmanFinder_H_
#define _ConfigurableTrimmedKalmanFinder_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableTrimmedKalmanFinder : public AbstractConfReconstructor
{
  public:
    ConfigurableTrimmedKalmanFinder ();
    void configure ( const edm::ParameterSet & ) override;
    ConfigurableTrimmedKalmanFinder ( const ConfigurableTrimmedKalmanFinder & o );
    ~ConfigurableTrimmedKalmanFinder() override;
    ConfigurableTrimmedKalmanFinder * clone () const override;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const override;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t,
        const reco::BeamSpot & s ) const override;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & prims,
        const std::vector < reco::TransientTrack > & secs,
        const reco::BeamSpot & s ) const override;
    edm::ParameterSet defaults() const override;
  private:
    const VertexReconstructor * theRector;
};

#endif
