#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonObjectProducer.h"

//
// constructors and destructor
//
TopMuonObjectProducer::TopMuonObjectProducer(const edm::ParameterSet& iConfig)
{
   muonPTcut_      = iConfig.getParameter< double > ("muonPTcut");
   muonEtacut_     = iConfig.getParameter< double > ("muonEtacut");
   muonLRcut_      = iConfig.getParameter< double > ("muonLRcut");
   addResolutions_ = iConfig.getParameter< bool >   ("addResolutions");
   
   //produces vector of muons
   produces<vector<TopMuonObject > >("muons");
}


TopMuonObjectProducer::~TopMuonObjectProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopMuonObjectProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   Handle<vector<muonType> > muons;
   iEvent.getByLabel("globalMuons", muons );

   //loop over muons
   vector<TopMuonObject> * ttMuons = new vector<TopMuonObject>(); 
   for(size_t m=0; m<muons->size(); m++){
     if( (*muons)[m].pt()>muonPTcut_ && fabs((*muons)[m].eta())<muonEtacut_ ){
       
       TopMuon aMuon((*muons)[m]);
       if(addResolutions_){
	 aMuon.setResET(5.);
	 aMuon.setResEta(0.0005);
	 aMuon.setResD(0.5);
	 aMuon.setResPhi(0.0003);
	 aMuon.setResTheta(0.0001);
	 aMuon.setResPinv(0.0002);
       }
       ttMuons->push_back(TopMuonObject(aMuon));
     }
   }
   // sort muons in pT
   std::sort(ttMuons->begin(),ttMuons->end(),pTMuonComparator);

   // put genEvt object in Event
   auto_ptr<vector<TopMuonObject> > pOutMuon(ttMuons);
   iEvent.put(pOutMuon,"muons");
}
