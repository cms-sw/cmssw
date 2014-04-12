#ifndef _ConfigurableMultiVertexFitter_H_
#define _ConfigurableMultiVertexFitter_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"

class MultiVertexReconstructor;

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableMultiVertexFitter : public AbstractConfReconstructor
{
  public:
    /**
     *
     * Accepted values:
     *  sigmacut: The sqrt(chi2_cut) criterion. Default: 3.0
     *  ratio:   The annealing ratio. Default: 0.25
     *  Tini:    The initial temparature. Default: 256
     *
     */
    ConfigurableMultiVertexFitter ();
    ConfigurableMultiVertexFitter ( const ConfigurableMultiVertexFitter & o );
    ~ConfigurableMultiVertexFitter();
    ConfigurableMultiVertexFitter * clone () const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t,
        const reco::BeamSpot & s ) const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & prims,
        const std::vector < reco::TransientTrack > & secs,
        const reco::BeamSpot & s ) const;
    void configure ( const edm::ParameterSet & );
    edm::ParameterSet defaults() const;
  private:
    const MultiVertexReconstructor * theRector;
    int theCheater;
};

#endif
