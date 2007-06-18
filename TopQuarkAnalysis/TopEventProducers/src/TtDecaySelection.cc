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

   // check the event decay
   edm::Handle<TtGenEvent>  genEvt;
   iEvent.getByLabel ("genEvt",genEvt);
   
   int channel  = abs( (int) (decay_*1./10.) );
   int allowLep = abs( (int) decay_ - channel*10 );   
   
   
   bool decision = false;
   
   // fully hadronic decay
   if(channel == 0 && genEvt->decay() == 0) decision = true;
   
   // semilep decay
   if(channel == 1 && genEvt->decay() == 1){
     if (allowLep == 0) decision = true; //no further demands on leptons
     if (allowLep == 1 &&   abs(genEvt->particles()[4]->pdgId()) == 11)  decision = true; // only electron
     if (allowLep == 2 &&   abs(genEvt->particles()[4]->pdgId()) == 13)  decision = true; // only muon
     if (allowLep == 3 &&   abs(genEvt->particles()[4]->pdgId()) == 15)  decision = true; // only tau
     if (allowLep == 4 && !(abs(genEvt->particles()[4]->pdgId()) == 11)) decision = true; // no electron
     if (allowLep == 5 && !(abs(genEvt->particles()[4]->pdgId()) == 13)) decision = true; // no muon
     if (allowLep == 6 && !(abs(genEvt->particles()[4]->pdgId()) == 15)) decision = true; // no tau
   }
   
   // fully leptonic decay
   if(channel == 2 && genEvt->decay() == 2){
     if (allowLep == 0) decision = true; //no further demands on leptons
     if (allowLep == 1 &&   abs(genEvt->particles()[0]->pdgId()) == 11  &&   abs(genEvt->particles()[4]->pdgId()) == 11)  decision = true; // only electron
     if (allowLep == 2 &&   abs(genEvt->particles()[0]->pdgId()) == 13  &&   abs(genEvt->particles()[4]->pdgId()) == 13)  decision = true; // only muon
     if (allowLep == 3 &&   abs(genEvt->particles()[0]->pdgId()) == 15  &&   abs(genEvt->particles()[4]->pdgId()) == 15)  decision = true; // only tau
     if (allowLep == 4 && !(abs(genEvt->particles()[0]->pdgId()) == 11) && !(abs(genEvt->particles()[4]->pdgId()) == 11)) decision = true; // no electron
     if (allowLep == 5 && !(abs(genEvt->particles()[0]->pdgId()) == 13) && !(abs(genEvt->particles()[4]->pdgId()) == 13)) decision = true; // no muon
     if (allowLep == 6 && !(abs(genEvt->particles()[0]->pdgId()) == 15) && !(abs(genEvt->particles()[4]->pdgId()) == 15)) decision = true; // no tau
   }
   
   if(decay_ < 0) decision = !decision;
   return decision;
   
}

