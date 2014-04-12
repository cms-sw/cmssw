#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerBase.h"

ParticleReplacerBase::ParticleReplacerBase(const edm::ParameterSet& cfg)
  : tried_(0), 
    passed_(0), 
    tauMass_(1.7769)
{
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

#include "FWCore/Framework/interface/MakerMacros.h"

EDM_REGISTER_PLUGINFACTORY(ParticleReplacerPluginFactory, "ParticleReplacerPluginFactory");
