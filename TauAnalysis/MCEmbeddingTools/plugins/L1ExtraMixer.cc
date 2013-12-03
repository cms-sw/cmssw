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
    std::string instanceLabel = cfgPlugin->getParameter<std::string>("instanceLabel");
    edm::InputTag src1withInstanceLabel(src1.label(), instanceLabel, src1.process());
    cfgPlugin->addParameter<edm::InputTag>("src1", src1withInstanceLabel);
    edm::InputTag src2withInstanceLabel(src2.label(), instanceLabel, src2.process());
    cfgPlugin->addParameter<edm::InputTag>("src2", src2withInstanceLabel);
    L1ExtraMixerPluginBase* plugin = L1ExtraMixerPluginFactory::get()->create(pluginType, *cfgPlugin);
    plugin->registerProducts(*this);
    plugins_.push_back(plugin);
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
  for ( std::vector<L1ExtraMixerPluginBase*>::iterator plugin = plugins_.begin();
	plugin != plugins_.end(); ++plugin ) {
    (*plugin)->produce(evt, es);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(L1ExtraMixer);
