#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include <vector>
#include <iostream>

typedef SingleObjectSelector<std::vector<reco::Jet>, StringCutObjectSelector<reco::Jet> > TauValJetSelector;
typedef SingleObjectSelector<reco::MuonCollection, StringCutObjectSelector<reco::Muon> > TauValMuonSelector;
typedef SingleObjectSelector<reco::GenParticleCollection, StringCutObjectSelector<reco::GenParticle> >
    TauValGenPSelector;
typedef SingleObjectSelector<reco::GenParticleRefVector, StringCutObjectSelector<reco::GenParticleRef> >
    TauValGenPRefSelector;
typedef SingleObjectSelector<reco::PFJetCollection, StringCutObjectSelector<reco::PFJet> > TauValPFJetSelector;
typedef SingleObjectSelector<edm::View<reco::GsfElectron>,
                             StringCutObjectSelector<reco::GsfElectron>,
                             reco::GsfElectronCollection>
    TauValElectronSelector;

DEFINE_FWK_MODULE(TauValPFJetSelector);
DEFINE_FWK_MODULE(TauValJetSelector);
DEFINE_FWK_MODULE(TauValMuonSelector);
DEFINE_FWK_MODULE(TauValElectronSelector);
DEFINE_FWK_MODULE(TauValGenPSelector);
DEFINE_FWK_MODULE(TauValGenPRefSelector);

class ElectronIdFilter : public edm::global::EDFilter<> {
public:
  explicit ElectronIdFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<reco::GsfElectronCollection> recoGsfElectronCollectionToken_;
  edm::EDGetTokenT<edm::ValueMap<float> > edmValueMapFloatToken_;
  int eid_;
  // ----------member data ---------------------------
};

ElectronIdFilter::ElectronIdFilter(const edm::ParameterSet& iConfig)
    : recoGsfElectronCollectionToken_(
          consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      edmValueMapFloatToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("eidsrc"))),
      eid_(iConfig.getParameter<int>("eid")) {
  produces<reco::GsfElectronCollection>();
}

// ------------ method called to produce the data  ------------

bool ElectronIdFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  //cout << "NonVertexingLeptonFilter:: entering filter" << endl;

  edm::Handle<reco::GsfElectronCollection> electrons;
  iEvent.getByToken(recoGsfElectronCollectionToken_, electrons);

  edm::Handle<edm::ValueMap<float> > eIDValueMap;
  iEvent.getByToken(edmValueMapFloatToken_, eIDValueMap);
  const edm::ValueMap<float>& eIDmap = *eIDValueMap;
  reco::GsfElectronCollection* product = new reco::GsfElectronCollection();

  // Loop over electrons
  for (unsigned int i = 0; i < electrons->size(); i++) {
    edm::Ref<reco::GsfElectronCollection> electronRef(electrons, i);
    if ((eIDmap[electronRef]) == eid_)
      product->push_back((*electrons)[i]);
  }

  //cout << "Putting in the event" << endl;
  std::unique_ptr<reco::GsfElectronCollection> collection(product);
  iEvent.put(std::move(collection));
  return true;
}

DEFINE_FWK_MODULE(ElectronIdFilter);
