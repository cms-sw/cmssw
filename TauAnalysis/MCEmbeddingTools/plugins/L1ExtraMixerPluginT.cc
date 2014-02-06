#include "TauAnalysis/MCEmbeddingTools/plugins/L1ExtraMixerPluginT.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <TMath.h>

const size_t maxNumL1ExtraObjects = 4;

template <typename T>
L1ExtraMixerPluginT<T>::L1ExtraMixerPluginT(const edm::ParameterSet& cfg)
  : L1ExtraMixerPluginBase(cfg)
{
  if ( cfg.exists("srcSelectedMuons1") ) {
    srcSelectedMuons1_ = cfg.getParameter<edm::InputTag>("srcSelectedMuons1");
    dRveto1_ = cfg.getParameter<double>("dRveto1");
  }
  if ( cfg.exists("srcSelectedMuons2") ) {
    srcSelectedMuons2_ = cfg.getParameter<edm::InputTag>("srcSelectedMuons2");
    dRveto2_ = cfg.getParameter<double>("dRveto2");
  }
}

template <typename T>
void L1ExtraMixerPluginT<T>::registerProducts(edm::EDProducer& producer)
{
  producer.produces<l1ExtraCollection>(instanceLabel_); 
}

namespace
{
  template <typename T>
  std::vector<const T*> getCleanedCollection(const std::vector<T>& l1ExtraObjects, const edm::Event& evt, const edm::InputTag& srcSelectedMuons, double dRveto)
  {
    std::vector<const T*> l1ExtraObjects_cleaned;
    if ( srcSelectedMuons.label() != "" ) {
      std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons);
      for ( typename std::vector<T>::const_iterator l1ExtraObject = l1ExtraObjects.begin();
	    l1ExtraObject != l1ExtraObjects.end(); ++l1ExtraObject ) {
	bool isVetoed = false;
	for ( std::vector<reco::CandidateBaseRef>::const_iterator selMuon = selMuons.begin();
	      selMuon != selMuons.end(); ++selMuon ) {
	  double dR = deltaR(l1ExtraObject->p4(), (*selMuon)->p4());
	  if ( dR < dRveto ) isVetoed = true;
	}
	if ( !isVetoed ) l1ExtraObjects_cleaned.push_back(&(*l1ExtraObject));
      }
    } else {
      for ( typename std::vector<T>::const_iterator l1ExtraObject = l1ExtraObjects.begin();
	    l1ExtraObject != l1ExtraObjects.end(); ++l1ExtraObject ) {
	l1ExtraObjects_cleaned.push_back(&(*l1ExtraObject));
      }
    }
    return l1ExtraObjects_cleaned;
  }

  template <typename TPtr>
  struct higherPtT
  {
    bool operator() (const TPtr& t1, const TPtr& t2)
    {
      return (t1->pt() > t2->pt());
    }
  };
}

template <typename T>
void L1ExtraMixerPluginT<T>::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<l1ExtraCollection> l1ExtraObjects1;
  evt.getByLabel(src1_, l1ExtraObjects1);
  l1ExtraPtrCollection l1ExtraObjects1_cleaned = getCleanedCollection(*l1ExtraObjects1, evt, srcSelectedMuons1_, dRveto1_);

  edm::Handle<l1ExtraCollection> l1ExtraObjects2;
  evt.getByLabel(src2_, l1ExtraObjects2);
  l1ExtraPtrCollection l1ExtraObjects2_cleaned = getCleanedCollection(*l1ExtraObjects2, evt, srcSelectedMuons2_, dRveto2_);
  
  l1ExtraPtrCollection l1ExtraObjects_sorted;
  l1ExtraObjects_sorted.insert(l1ExtraObjects_sorted.end(), l1ExtraObjects1_cleaned.begin(), l1ExtraObjects1_cleaned.end());
  l1ExtraObjects_sorted.insert(l1ExtraObjects_sorted.end(), l1ExtraObjects2_cleaned.begin(), l1ExtraObjects2_cleaned.end());
  higherPtT<const T*> higherPt;
  std::sort(l1ExtraObjects_sorted.begin(), l1ExtraObjects_sorted.end(), higherPt);

  std::auto_ptr<l1ExtraCollection> l1ExtraObjects_output(new l1ExtraCollection());

  size_t numL1ExtraObjects = l1ExtraObjects_sorted.size();
  for ( size_t iObject = 0; iObject < TMath::Min(numL1ExtraObjects, maxNumL1ExtraObjects); ++iObject ) {
    l1ExtraObjects_output->push_back(*l1ExtraObjects_sorted.at(iObject));
  }

  evt.put(l1ExtraObjects_output, instanceLabel_);
}

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

typedef L1ExtraMixerPluginT<l1extra::L1EmParticle> L1ExtraEmParticleMixerPlugin;
typedef L1ExtraMixerPluginT<l1extra::L1MuonParticle> L1ExtraMuonParticleMixerPlugin;
typedef L1ExtraMixerPluginT<l1extra::L1JetParticle> L1ExtraJetParticleMixerPlugin;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(L1ExtraMixerPluginFactory, L1ExtraEmParticleMixerPlugin, "L1ExtraEmParticleMixerPlugin");
DEFINE_EDM_PLUGIN(L1ExtraMixerPluginFactory, L1ExtraMuonParticleMixerPlugin, "L1ExtraMuonParticleMixerPlugin");
DEFINE_EDM_PLUGIN(L1ExtraMixerPluginFactory, L1ExtraJetParticleMixerPlugin, "L1ExtraJetParticleMixerPlugin");
