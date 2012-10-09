#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerFactory.h"

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerClass.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerParticleGun.h"

boost::shared_ptr<ParticleReplacerBase> ParticleReplacerFactory::create(const std::string& algo, const edm::ParameterSet& cfg) 
{
  if ( algo == "ZTauTau" ) return boost::shared_ptr<ParticleReplacerBase>(new ParticleReplacerClass(cfg.getParameter<edm::ParameterSet>("ZTauTau")));
  else if ( algo == "ParticleGun" ) return boost::shared_ptr<ParticleReplacerBase>(new ParticleReplacerParticleGun(cfg.getParameter<edm::ParameterSet>("ParticleGun")));
  else throw cms::Exception("Configuration") 
    << "Unknown particle replacer algorithm = " << algo << " (supported algorithms: 'ZTauTau', 'ParticleGun') !!\n";
}
