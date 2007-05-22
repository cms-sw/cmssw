#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"

//
// constructors and destructor
//
TtGenEventReco::TtGenEventReco(const edm::ParameterSet& iConfig)
{
   produces<TtGenEvent>();
}


TtGenEventReco::~TtGenEventReco()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TtGenEventReco::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   edm::Handle<reco::CandidateCollection> genParticles;
   iEvent.getByLabel( "genParticleCandidates", genParticles );
   if(genParticles->size() == 0) cout<<"No GenParticle Candidates found..."<<endl;  
   
   // search top quarks
   vector<const reco::Candidate *> tvec;
   for( size_t p=0; (p<genParticles->size() && tvec.size()<2); p++) {
     if(status((*genParticles)[p]) == 3 && abs((*genParticles)[p].pdgId()) == 6) tvec.push_back(&((*genParticles)[p]));
   }
   //cout<<"found "<<tvec.size()<<" top quarks..."<<endl; 
   if(tvec.size()<2) cout<<"No ttbar process!"<<endl;  
  
   
   // search W-bosons and b-quarks
   vector<const reco::Candidate *> Wvec, bvec;
   if(tvec.size() == 2){
     for( size_t t=0; t<tvec.size(); t++) {
       for( size_t d=0; d<tvec[t]->numberOfDaughters(); d++) {
         if(status(*tvec[t]->daughter(d)) == 3 && abs(tvec[t]->daughter(d)->pdgId()) == 24) Wvec.push_back(tvec[t]->daughter(d));
         if(status(*tvec[t]->daughter(d)) == 3 && abs(tvec[t]->daughter(d)->pdgId()) ==  5) bvec.push_back(tvec[t]->daughter(d));
       }
     }
   }
   //cout<<"found "<<Wvec.size()<<" W bosons..."<<endl;   
   //cout<<"found "<<bvec.size()<<" b quarks..."<<endl;   
   
   
   // search W-decay products
   int Whadr = 0;
   vector<const reco::Candidate *> qvec, lvec, nvec;
   if(Wvec.size() == 2 && bvec.size() == 2){
     for( size_t w=0; w<tvec.size(); w++) {
       for( size_t d=0; d<Wvec[w]->numberOfDaughters(); d++) {
         if(status(*Wvec[w]->daughter(d)) == 3 && abs(Wvec[w]->daughter(d)->pdgId()) < 5) {qvec.push_back(Wvec[w]->daughter(d)); Whadr = w;};
         if(status(*Wvec[w]->daughter(d)) == 3 && abs(Wvec[w]->daughter(d)->pdgId()) ==  11 || abs(Wvec[w]->daughter(d)->pdgId()) ==  13 || abs(Wvec[w]->daughter(d)->pdgId()) ==  15) lvec.push_back(Wvec[w]->daughter(d));
         if(status(*Wvec[w]->daughter(d)) == 3 && abs(Wvec[w]->daughter(d)->pdgId()) ==  12 || abs(Wvec[w]->daughter(d)->pdgId()) ==  14 || abs(Wvec[w]->daughter(d)->pdgId()) ==  16) nvec.push_back(Wvec[w]->daughter(d));
       }
     }
   } 
   //cout<<"found "<<qvec.size()<<" light quarks..."<<endl; 
   //cout<<"found "<<lvec.size()<<" charged leptons..."<<endl; 
   //cout<<"found "<<nvec.size()<<" neutrinos..."<<endl; 


   // fill TtGenEvent object depending on decay
   int decay = -999;
   vector<const reco::Candidate *> evtvec;
   //  semilep
   if(qvec.size() == 2 && lvec.size() == 1 && nvec.size() == 1){
     decay = 1;
     evtvec.push_back(qvec[0]); 
     evtvec.push_back(qvec[1]); 
     evtvec.push_back(bvec[Whadr]); 
     evtvec.push_back(bvec[1-Whadr]); 
     evtvec.push_back(lvec[0]);  
     evtvec.push_back(nvec[0]); 
     evtvec.push_back(Wvec[Whadr]);
     evtvec.push_back(Wvec[1-Whadr]);
     evtvec.push_back(tvec[Whadr]);
     evtvec.push_back(tvec[1-Whadr]);
   }	 
   //fullylep
   else if(qvec.size() == 0 && lvec.size() == 2 && nvec.size() == 2){
     decay = 2;
     evtvec.push_back(lvec[0]); 
     evtvec.push_back(nvec[0]); 
     evtvec.push_back(bvec[0]); 
     evtvec.push_back(bvec[1]); 
     evtvec.push_back(lvec[1]);  
     evtvec.push_back(nvec[1]); 
     evtvec.push_back(Wvec[0]);
     evtvec.push_back(Wvec[1]);
     evtvec.push_back(tvec[0]);
     evtvec.push_back(tvec[1]);
   }	
   //fully hadr
   else if(qvec.size() == 4 && lvec.size() == 0 && nvec.size() == 0){
     decay = 0;
     evtvec.push_back(qvec[0]); 
     evtvec.push_back(qvec[1]); 
     evtvec.push_back(bvec[0]); 
     evtvec.push_back(bvec[1]); 
     evtvec.push_back(qvec[2]);  
     evtvec.push_back(qvec[3]); 
     evtvec.push_back(Wvec[0]);
     evtvec.push_back(Wvec[1]);
     evtvec.push_back(tvec[0]);
     evtvec.push_back(tvec[1]);
   }
   else {
     cout<<"no semilep, fullylep or fullyhadr decay???"<<endl;
   }
   
   // put genEvt object in Event
   TtGenEvent * genEvt = new TtGenEvent(decay,evtvec);
   auto_ptr<TtGenEvent> myTtGenEvent(genEvt);
   iEvent.put(myTtGenEvent);
   

}
