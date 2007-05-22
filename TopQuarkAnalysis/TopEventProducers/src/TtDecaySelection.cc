#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySelection.h"

//
// constructors and destructor
//
TtDecaySelection::TtDecaySelection(const edm::ParameterSet& iConfig)
{
   decay_  = iConfig.getParameter< int > ("allowDecay");
   std::cout << "TtDecaySelection allowDecay: " << decay_ << std::endl;
}


TtDecaySelection::~TtDecaySelection()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool TtDecaySelection::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   if(decay_ == -1) return true;   

   // check the event decay
   edm::Handle<TtGenEvent>  genEvt;
   iEvent.getByLabel ("genEvt",genEvt);
   
   int channel  = (int) (decay_*1./10.);
   int allowLep = (int) decay_ - channel*10;   
   
   // fully hadronic decay
   if(channel == 0 && genEvt->decay() == 0) return true;
   
   // semilep decay
   if(channel == 1 && genEvt->decay() == 1){
     if (allowLep == 0) return true; //no further demands on leptons
     if (allowLep == 1 &&   abs(genEvt->particles()[4]->pdgId()) == 11)  return true; // only electron
     if (allowLep == 2 &&   abs(genEvt->particles()[4]->pdgId()) == 13)  return true; // only muon
     if (allowLep == 3 &&   abs(genEvt->particles()[4]->pdgId()) == 15)  return true; // only tau
     if (allowLep == 4 && !(abs(genEvt->particles()[4]->pdgId()) == 11)) return true; // no electron
     if (allowLep == 5 && !(abs(genEvt->particles()[4]->pdgId()) == 13)) return true; // no muon
     if (allowLep == 6 && !(abs(genEvt->particles()[4]->pdgId()) == 15)) return true; // no tau
   }
   
   // fully leptonic decay
   if(channel == 2 && genEvt->decay() == 2){
     if (allowLep == 0) return true; //no further demands on leptons
     if (allowLep == 1 &&   abs(genEvt->particles()[0]->pdgId()) == 11  &&   abs(genEvt->particles()[4]->pdgId()) == 11)  return true; // only electron
     if (allowLep == 2 &&   abs(genEvt->particles()[0]->pdgId()) == 13  &&   abs(genEvt->particles()[4]->pdgId()) == 13)  return true; // only muon
     if (allowLep == 3 &&   abs(genEvt->particles()[0]->pdgId()) == 15  &&   abs(genEvt->particles()[4]->pdgId()) == 15)  return true; // only tau
     if (allowLep == 4 && !(abs(genEvt->particles()[0]->pdgId()) == 11) && !(abs(genEvt->particles()[4]->pdgId()) == 11)) return true; // no electron
     if (allowLep == 5 && !(abs(genEvt->particles()[0]->pdgId()) == 13) && !(abs(genEvt->particles()[4]->pdgId()) == 13)) return true; // no muon
     if (allowLep == 6 && !(abs(genEvt->particles()[0]->pdgId()) == 15) && !(abs(genEvt->particles()[4]->pdgId()) == 15)) return true; // no tau
   }
   
   return false;
}

