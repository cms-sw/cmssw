#ifndef _ConfigurableAdaptiveFitter_H_
#define _ConfigurableAdaptiveFitter_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableAdaptiveFitter : public AbstractConfReconstructor
{
  public:
    /**
     *  Values that are respected:
     *  sigmacut: The sqrt(chi2_cut) criterion. Default: 3.0
     *  ratio:   The annealing ratio. Default: 0.25
     *  Tini:    The initial temparature. Default: 256
     */
    ConfigurableAdaptiveFitter ();
    void configure ( const edm::ParameterSet & );
    ConfigurableAdaptiveFitter ( const ConfigurableAdaptiveFitter & o );
    ~ConfigurableAdaptiveFitter();
    ConfigurableAdaptiveFitter * clone () const;
    std::vector < TransientVertex > vertices ( 
        const std::vector < reco::TransientTrack > & t ) const;
    edm::ParameterSet defaults() const;
  private:
    const VertexReconstructor * theRector;
};

#endif
