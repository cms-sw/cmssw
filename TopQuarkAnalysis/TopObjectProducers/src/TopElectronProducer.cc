#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"


//
// constructors and destructor
//
TopElectronProducer::TopElectronProducer(const edm::ParameterSet& iConfig)
{
   electronPTcut_  	= iConfig.getParameter< double > ("electronPTcut");
   electronEtacut_ 	= iConfig.getParameter< double > ("electronEtacut");
   electronLRcut_  	= iConfig.getParameter< double > ("electronLRcut");
   addResolutions_ 	= iConfig.getParameter< bool   > ("addResolutions");
   addLRValues_    	= iConfig.getParameter<bool>("addLRValues");
   electronLRFile_ 	= iConfig.getParameter<string>("electronLRFile");
   electronResoFile_	= iConfig.getParameter<string>("electronResoFile");
   
   //construct resolution calculator
   if(addResolutions_){
     elResCalc = new TopObjectResolutionCalc(electronResoFile_);
   }

   //produces vector of electrons
   produces<vector<TopElectron > >("electrons");
}


TopElectronProducer::~TopElectronProducer()
{
   if(addResolutions_) delete elResCalc;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   if (addLRValues_) {
     theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, electronLRFile_, "");
   }

   // Get the vector of generated particles from the event
   edm::Handle<vector<ElectronType> > electrons; 
   iEvent.getByLabel("pixelMatchGsfElectrons", electrons );
   
   //loop over electrons
   vector<TopElectron> * topElectrons = new vector<TopElectron>(); 
   for(size_t e=0; e<electrons->size(); e++){
     if( (*electrons)[e].pt()>electronPTcut_ && fabs((*electrons)[e].eta())<electronEtacut_ ){

       TopElectron anElectron((*electrons)[e]);
       // add resolution info if demanded
       if(addResolutions_){
         (*elResCalc)(anElectron);
       }
       // add top lepton id LR info if requested
       if (addLRValues_) {
         theLeptonLRCalc_->calcLikelihood(anElectron, iEvent);
       }
       topElectrons->push_back(TopElectron(anElectron));
     }
   }
   // sort electrons in pT
   std::sort(topElectrons->begin(),topElectrons->end(),pTElectronComparator);

   if (addLRValues_) delete theLeptonLRCalc_;

   // put genEvt object in Event
   auto_ptr<vector<TopElectron> > pOutElectron(topElectrons);
   iEvent.put(pOutElectron,"electrons");
}
