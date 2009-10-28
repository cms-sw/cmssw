#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerFactory.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerClass.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerParticleGun.h"

boost::shared_ptr<ParticleReplacerBase> ParticleReplacerFactory::create(const std::string& algo, const edm::ParameterSet& iConfig) {
  bool verbose = iConfig.getParameter<bool>("verbose");
  if(algo == "ZTauTau")
    return boost::shared_ptr<ParticleReplacerBase>(new ParticleReplacerClass(iConfig.getParameter<edm::ParameterSet>("ZTauTau"), verbose));
  else if(algo == "ParticleGun")
    return boost::shared_ptr<ParticleReplacerBase>(new ParticleReplacerParticleGun(iConfig.getParameter<edm::ParameterSet>("ParticleGun"), verbose));
  else 
    throw cms::Exception("Configuration") << "Unknown particle replacer algorithm " << algo
                                          << ". Supported algorithms: 'ZTauTau', 'ParticleGun'." << std::endl;
}
