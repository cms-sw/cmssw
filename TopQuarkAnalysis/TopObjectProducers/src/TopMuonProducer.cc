#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"


//
// constructors and destructor
//
TopMuonProducer::TopMuonProducer(const edm::ParameterSet& iConfig)
{
   muonPTcut_      = iConfig.getParameter< double > ("muonPTcut");
   muonEtacut_     = iConfig.getParameter< double > ("muonEtacut");
   muonLRcut_      = iConfig.getParameter< double > ("muonLRcut");
   addResolutions_ = iConfig.getParameter< bool   > ("addResolutions");
   addLRValues_    = iConfig.getParameter<bool>("addLRValues");
   muonLRFile_     = iConfig.getParameter<string>("muonLRFile");
   muonResoFile_   = iConfig.getParameter<string>("muonResoFile");
   
   //construct resolution calculator
   if(addResolutions_){
     muResCalc = new TopObjectResolutionCalc(muonResoFile_);
   }
   
   //produces vector of muons
   produces<vector<TopMuon > >("muons");
}


TopMuonProducer::~TopMuonProducer()
{
   if(addResolutions_) delete muResCalc;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   if (addLRValues_) {
     theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, "", muonLRFile_);
   }
  
   // Get the vector of generated particles from the event
   edm::Handle<vector<MuonType> > muons;
   iEvent.getByLabel("globalMuons", muons );

   //loop over muons
   vector<TopMuon> * topMuons = new vector<TopMuon>(); 
   for(size_t m=0; m<muons->size(); m++){
     if( (*muons)[m].pt()>muonPTcut_ && fabs((*muons)[m].eta())<muonEtacut_ ){
       
       TopMuon aMuon((*muons)[m]);
       // add resolution info if demanded
       if(addResolutions_){
         (*muResCalc)(aMuon);
       }
       // add top lepton id LR info if requested
       if (addLRValues_) {
         theLeptonLRCalc_->calcLikelihood(aMuon, iEvent);
       }
       topMuons->push_back(TopMuon(aMuon));
     }
   }
   // sort muons in pT
   std::sort(topMuons->begin(),topMuons->end(),pTMuonComparator);

   if (addLRValues_) delete theLeptonLRCalc_;

   // put genEvt object in Event
   auto_ptr<vector<TopMuon> > pOutMuon(topMuons);
   iEvent.put(pOutMuon,"muons");
}
