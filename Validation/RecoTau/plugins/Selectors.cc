#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

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

typedef SingleObjectSelector< std::vector<reco::Jet>        , StringCutObjectSelector<reco::Jet>            >   TauValJetSelector     ;
typedef SingleObjectSelector< reco::MuonCollection          , StringCutObjectSelector<reco::Muon>           >   TauValMuonSelector    ;
typedef SingleObjectSelector< reco::GenParticleCollection   , StringCutObjectSelector<reco::GenParticle>    >   TauValGenPSelector    ;
typedef SingleObjectSelector< reco::GenParticleRefVector    , StringCutObjectSelector<reco::GenParticleRef> >   TauValGenPRefSelector ;
typedef SingleObjectSelector< reco::PFJetCollection         , StringCutObjectSelector<reco::PFJet>          >   TauValPFJetSelector   ;
typedef SingleObjectSelector< edm::View<reco::GsfElectron>  , StringCutObjectSelector<reco::GsfElectron>, reco::GsfElectronCollection > TauValElectronSelector;

DEFINE_FWK_MODULE( TauValPFJetSelector );
DEFINE_FWK_MODULE( TauValJetSelector );
DEFINE_FWK_MODULE( TauValMuonSelector );
DEFINE_FWK_MODULE( TauValElectronSelector );
DEFINE_FWK_MODULE( TauValGenPSelector );
DEFINE_FWK_MODULE( TauValGenPRefSelector );

class ElectronIdFilter : public edm::EDFilter {
public:
  explicit ElectronIdFilter(const edm::ParameterSet&);
  ~ElectronIdFilter();

private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
      
  edm::InputTag src_, eidsrc_;
  int eid_; 
  // ----------member data ---------------------------
};

ElectronIdFilter::ElectronIdFilter(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  eidsrc_(iConfig.getParameter<edm::InputTag>("eidsrc")),
  eid_(iConfig.getParameter<int>("eid"))
{
  produces< reco::GsfElectronCollection >();
}
ElectronIdFilter::~ElectronIdFilter()
{
}
// ------------ method called to produce the data  ------------

bool
ElectronIdFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //cout << "NonVertexingLeptonFilter:: entering filter" << endl;
  using namespace reco;
  using namespace std;
  
  edm::Handle<GsfElectronCollection> electrons;
  iEvent.getByLabel(src_,electrons);
  
  edm::Handle<edm::ValueMap<float> > eIDValueMap;
  iEvent.getByLabel(eidsrc_, eIDValueMap);
  const edm::ValueMap<float> & eIDmap = * eIDValueMap;
  GsfElectronCollection *product = new GsfElectronCollection();

  // Loop over electrons
  for (unsigned int i = 0; i < electrons->size(); i++){
    edm::Ref<reco::GsfElectronCollection> electronRef(electrons,i);
    if((eIDmap[electronRef]) == eid_)
      product->push_back((*electrons)[i]);
  }

  //cout << "Putting in the event" << endl;
  std::auto_ptr<GsfElectronCollection> collection(product);
  iEvent.put(collection);
  return true;
}

void 
ElectronIdFilter::beginJob() {
}
void 
ElectronIdFilter::endJob() {
}

DEFINE_FWK_MODULE(ElectronIdFilter);
