#include "Validation/RecoTau/interface/GenJetRefProducer.h"
#include "TLorentzVector.h"


using namespace edm;
using namespace std;

GenJetRefProducer::GenJetRefProducer(const edm::ParameterSet& mc)
{
 
  //One Parameter Set per Collection

  genJetSrc_      = mc.getUntrackedParameter<edm::InputTag>("GenJetSrc");
  ptMinGenJet_ = mc.getUntrackedParameter<double>("ptMinGenJet",5.);
  etaMax = mc.getUntrackedParameter<double>("EtaMax",2.5);

  produces<LorentzVectorCollection>("GenJets");
  
}

GenJetRefProducer::~GenJetRefProducer(){ }

void GenJetRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{
  //All the code from HLTTauMCInfo is here :-) 
  
  auto_ptr<LorentzVectorCollection> product_GenJets(new LorentzVectorCollection);
  
  
  edm::Handle< reco::GenJetCollection > genJets ;
  iEvent.getByLabel( genJetSrc_, genJets ) ;
  

  reco::GenJetCollection::const_iterator jetItr = genJets->begin();

  for ( ; jetItr != genJets->end(); jetItr++) {

    //    LorentzVector currentGenJet((*jetItr).px(),(*jetItr).py(),(*jetItr).pz(),(*jetItr).energy());
    LorentzVector currentGenJet(0.,0.,0.,0.);
    
    if ( (fabs( (*jetItr).eta()) < etaMax) && ((*jetItr).pt() > ptMinGenJet_ ) ) {
      currentGenJet.SetPxPyPzE((*jetItr).px(),(*jetItr).py(),(*jetItr).pz(),(*jetItr).energy());
      product_GenJets->push_back(currentGenJet);
    }
  }
    
  iEvent.put(product_GenJets,"GenJets");

}

