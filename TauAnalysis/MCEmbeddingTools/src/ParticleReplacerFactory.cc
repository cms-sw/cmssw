#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerFactory.h"

#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerClass.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerParticleGun.h"

boost::shared_ptr<ParticleReplacerBase> ParticleReplacerFactory::create(const std::string& algo, const edm::ParameterSet& cfg) 
{
  if ( algo == "ZTauTau" ) {
    edm::ParameterSet cfgParticleReplacer = cfg.getParameter<edm::ParameterSet>("ZTauTau");
    cfgParticleReplacer.addParameter<double>("beamEnergy", cfg.getParameter<double>("beamEnergy"));
    return boost::shared_ptr<ParticleReplacerBase>(new ParticleReplacerClass(cfgParticleReplacer));
  } else if ( algo == "ParticleGun" ) {
    edm::ParameterSet cfgParticleReplacer = cfg.getParameter<edm::ParameterSet>("ParticleGun");
    return boost::shared_ptr<ParticleReplacerBase>(new ParticleReplacerParticleGun(cfgParticleReplacer));
  } else throw cms::Exception("Configuration") 
      << "Unknown particle replacer algorithm = " << algo << " (supported algorithms: 'ZTauTau', 'ParticleGun') !!\n";
}
