#include "TauAnalysis/MCEmbeddingTools/plugins/L1ExtraMEtMixerPlugin.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/Common/interface/Handle.h"

L1ExtraMEtMixerPlugin::L1ExtraMEtMixerPlugin(const edm::ParameterSet& cfg)
  : L1ExtraMixerPluginBase(cfg)
{}

void L1ExtraMEtMixerPlugin::registerProducts(edm::EDProducer& producer)
{
  producer.produces<l1extra::L1EtMissParticleCollection>(instanceLabel_); 
}

void L1ExtraMEtMixerPlugin::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<l1extra::L1EtMissParticleCollection> met1;
  evt.getByLabel(src1_, met1);
  
  edm::Handle<l1extra::L1EtMissParticleCollection> met2;
  evt.getByLabel(src2_, met2);

  // CV: keep code general and do not assume 
  //     that there is exactly one MET object per event.
  //     The number of objects stored in the 'src1' and 'src2' do need to match, however,
  //     in order for the MET to be added vectorially
  if ( met1->size() != met2->size() )
    throw cms::Exception("L1ExtraMEtMixer::produce")
      << " Mismatch in numbers of MET objects stored in collections 'src1' and 'src2' !!\n";
  
  std::auto_ptr<l1extra::L1EtMissParticleCollection> metSum(new l1extra::L1EtMissParticleCollection());

  size_t numMETs = met1->size();
  for ( size_t iMET = 0; iMET < numMETs; ++iMET ) {
    const l1extra::L1EtMissParticle& met1_i = met1->at(iMET);
    const l1extra::L1EtMissParticle& met2_i = met2->at(iMET);

    // CV: check that both MET objects are of the same type
    if ( met1_i.type() != met2_i.type() )
      throw cms::Exception("L1ExtraMEtMixer::produce")
	<< " Mismatch in type between MET objects stored in collections 'src1' and 'src2' !!\n";

    // CV: setting edm::Refs to L1Gct objects not implemented yet
    l1extra::L1EtMissParticle metSum_i(
      met1_i.p4() + met2_i.p4(),
      met1_i.type(),
      met1_i.etTotal() + met2_i.etTotal());

    metSum->push_back(metSum_i);					       
  }

  evt.put(metSum, instanceLabel_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(L1ExtraMixerPluginFactory, L1ExtraMEtMixerPlugin, "L1ExtraMEtMixerPlugin");

