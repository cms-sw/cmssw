#include "TauAnalysis/MCEmbeddingTools/plugins/L1ExtraMixer.h"

L1ExtraMixer::L1ExtraMixer(const edm::ParameterSet& cfg)
{
  edm::InputTag src1 = cfg.getParameter<edm::InputTag>("src1");
  edm::InputTag src2 = cfg.getParameter<edm::InputTag>("src2");

  typedef std::vector<edm::ParameterSet> vParameterSet;
  vParameterSet cfgPlugins = cfg.getParameter<vParameterSet>("collections");
  for ( vParameterSet::iterator cfgPlugin = cfgPlugins.begin();
	cfgPlugin != cfgPlugins.end(); ++cfgPlugin ) {
    std::string pluginType = cfgPlugin->getParameter<std::string>("pluginType");
    cfgPlugin->addParameter<edm::InputTag>("src1", src1);
    cfgPlugin->addParameter<edm::InputTag>("src2", src2);
    plugins_.push_back(L1ExtraMixerPluginFactory::get()->create(pluginType, *cfgPlugin));
  }
}

L1ExtraMixer::~L1ExtraMixer()
{
  for ( std::vector<L1ExtraMixerPluginBase*>::iterator it = plugins_.begin();
	it != plugins_.end(); ++it ) {
    delete (*it);
  }
}

void L1ExtraMixer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::cout << "<L1ExtraMixer::produce>:" << std::endl;
  std::cout << " #plugins = " << plugins_.size() << std::endl;
  for ( std::vector<L1ExtraMixerPluginBase*>::iterator plugin = plugins_.begin();
	plugin != plugins_.end(); ++plugin ) {
    (*plugin)->produce(evt, es);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(L1ExtraMixer);
