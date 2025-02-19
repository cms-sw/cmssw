#ifndef _ConfigurableKalmanFitter_H_
#define _ConfigurableKalmanFitter_H_

#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"

/**
 *  Kalman filter, configurable version
 */

class ConfigurableKalmanFitter : public AbstractConfFitter
{
  public:
    ConfigurableKalmanFitter ();
    void configure ( const edm::ParameterSet & );
    ConfigurableKalmanFitter ( const ConfigurableKalmanFitter & o );
    ~ConfigurableKalmanFitter();
    ConfigurableKalmanFitter * clone () const;
    edm::ParameterSet defaults() const;

};

#endif
