#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronObjectProducer.h"

//
// constructors and destructor
//
TopElectronObjectProducer::TopElectronObjectProducer(const edm::ParameterSet& iConfig)
{
   electronPTcut_    = iConfig.getParameter< double > ("electronPTcut");
   electronEtacut_   = iConfig.getParameter< double > ("electronEtacut");
   electronLRcut_    = iConfig.getParameter< double > ("electronLRcut");
   addResolutions_   = iConfig.getParameter< bool   > ("addResolutions");
   
   //produces vector of electrons
   produces<vector<TopElectronObject > >("electrons");
}


TopElectronObjectProducer::~TopElectronObjectProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopElectronObjectProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   Handle<vector<electronType> > electrons; 
   iEvent.getByLabel("pixelMatchGsfElectrons", electrons );
   
   
   //loop over electrons
   vector<TopElectronObject> * ttElectrons = new vector<TopElectronObject>(); 
   for(size_t e=0; e<electrons->size(); e++){
     if( (*electrons)[e].pt()>electronPTcut_ && fabs((*electrons)[e].eta())<electronEtacut_ ){
       
       TopElectron anElectron((*electrons)[e]);
       // add resolution info if demanded
       if(addResolutions_){
	 anElectron.setResET(5.);
	 anElectron.setResEta(0.0005);
	 anElectron.setResD(0.5);
	 anElectron.setResPhi(0.0003);
	 anElectron.setResTheta(0.0001);
	 anElectron.setResPinv(0.0002);
       }
       ttElectrons->push_back(TopElectronObject(anElectron));
     }
   }
   // sort electrons in pT
   std::sort(ttElectrons->begin(),ttElectrons->end(),pTElectronComparator);

   // put genEvt object in Event
   auto_ptr<vector<TopElectronObject> > pOutElectron(ttElectrons);
   iEvent.put(pOutElectron,"electrons");
}
