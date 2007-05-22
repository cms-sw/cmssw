#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"

//
// constructors and destructor
//
StGenEventReco::StGenEventReco(const edm::ParameterSet& iConfig)
{
   produces<StGenEvent>();
}


StGenEventReco::~StGenEventReco()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
StGenEventReco::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   edm::Handle<reco::CandidateCollection> genParticles;
   iEvent.getByLabel( "genParticleCandidates", genParticles );
   if(genParticles->size() == 0) cout<<"No GenParticle Candidates found..."<<endl;  
   
   // search top quarks
   vector<const reco::Candidate *> tvec;
   for( size_t p=0; (p<genParticles->size() && tvec.size()<1); p++) {
     if(status((*genParticles)[p]) == 3 && abs((*genParticles)[p].pdgId()) == 6) tvec.push_back(&((*genParticles)[p]));
   }
   cout<<"found "<<tvec.size()<<" top quarks..."<<endl; 
   /*
   if(tvec.size()==2) cout<<"ttbar process!"<<endl;  
   if(tvec.size()==1) cout<<"single top process!"<<endl;   // temp
   if(tvec.size()==0) cout<<"non top process!"<<endl;  
   */
   
   // search W-bosons and b-quarks coming from top
   vector<const reco::Candidate *> Wvec, bvec;
   if(tvec.size() == 1){
     for( size_t t=0; t<tvec.size(); t++) {
       for( size_t d=0; d<tvec[t]->numberOfDaughters(); d++) {
         if(status(*tvec[t]->daughter(d)) == 3 && abs(tvec[t]->daughter(d)->pdgId()) == 24) Wvec.push_back(tvec[t]->daughter(d));
         if(status(*tvec[t]->daughter(d)) == 3 && abs(tvec[t]->daughter(d)->pdgId()) ==  5) bvec.push_back(tvec[t]->daughter(d));
       }
     }
   }
   cout<<"found "<<Wvec.size()<<" W bosons..."<<endl;   
   cout<<"found "<<bvec.size()<<" b quarks..."<<endl;   
   
   
   // search W-decay products
   vector<const reco::Candidate *> lvec, nvec;
   if(Wvec.size() ==1 && bvec.size() ==1){
     for( size_t w=0; w<tvec.size(); w++) {
       for( size_t d=0; d<Wvec[w]->numberOfDaughters(); d++) {
	 //         if(status(*Wvec[w]->daughter(d)) == 3 && abs(Wvec[w]->daughter(d)->pdgId()) < 5) {qvec.push_back(Wvec[w]->daughter(d)); Whadr = w;};
         if(status(*Wvec[w]->daughter(d)) == 3 && abs(Wvec[w]->daughter(d)->pdgId()) ==  11 || abs(Wvec[w]->daughter(d)->pdgId()) ==  13 || abs(Wvec[w]->daughter(d)->pdgId()) ==  15) lvec.push_back(Wvec[w]->daughter(d));
         if(status(*Wvec[w]->daughter(d)) == 3 && abs(Wvec[w]->daughter(d)->pdgId()) ==  12 || abs(Wvec[w]->daughter(d)->pdgId()) ==  14 || abs(Wvec[w]->daughter(d)->pdgId()) ==  16) nvec.push_back(Wvec[w]->daughter(d));
       }
     }
   } 
   cout<<"found "<<lvec.size()<<" charged leptons..."<<endl; 
   cout<<"found "<<nvec.size()<<" neutrinos..."<<endl; 

   // search recoil quark:
   vector<const reco::Candidate *> qvec;
   if (Wvec.size() ==1 && tvec.size() ==1) {
     for( size_t p=0; p<genParticles->size(); p++) {
       if(status((*genParticles)[p]) == 3 && abs((*genParticles)[p].pdgId()) <4) {
	 // check that it's not a daughter of a W:
	 bool veto=false;
	 for( size_t d=0; d<Wvec[0]->numberOfDaughters(); d++) {
	   if (Wvec[0]->daughter(d)==&((*genParticles)[p])) { veto=true; cout << "VETO!" << endl; }
	 }
	 if (!veto) qvec.push_back(&((*genParticles)[p]));
       }
     }
   }
   cout<<"found "<<qvec.size()<<" light quarks..."<<endl; 


   // fill StGenEvent object depending on decay
   int decay = -999;
   vector<const reco::Candidate *> evtvec;
   //  only one possibility considered so far
   if(qvec.size() > 0 && bvec.size() > 0 && lvec.size() == 1 && nvec.size() == 1){
     decay = 1;
     evtvec.push_back(bvec[0]); 
     evtvec.push_back(qvec[0]); 
     evtvec.push_back(lvec[0]);  
     evtvec.push_back(nvec[0]); 
     evtvec.push_back(Wvec[0]);
     evtvec.push_back(tvec[0]);
   }	 
   else {
     cout<<"no leptonic decay???"<<endl;
   }
   
   // put genEvt object in Event
   StGenEvent * genEvt = new StGenEvent(decay,evtvec);
   auto_ptr<StGenEvent> myStGenEvent(genEvt);
   iEvent.put(myStGenEvent);
   

}
