#ifndef Validation_RecoTrack_MTVHistoProducerAlgoFactory_h
#define Validation_RecoTrack_MTVHistoProducerAlgoFactory_h

#include "Validation/RecoTrack/interface/MTVHistoProducerAlgo.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet;}

 
typedef edmplugin::PluginFactory< MTVHistoProducerAlgo*(const edm::ParameterSet&) > MTVHistoProducerAlgoFactory;

#endif
