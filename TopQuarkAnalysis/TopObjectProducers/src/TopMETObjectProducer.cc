#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETObjectProducer.h"

//
// constructors and destructor
//
TopMETObjectProducer::TopMETObjectProducer(const edm::ParameterSet& iConfig)
{
   METLabel_   	   = iConfig.getParameter< string > ("METInput");
   METcut_         = iConfig.getParameter< double > ("METcut");
   addResolutions_ = iConfig.getParameter< bool   > ("addResolutions");
   produces<vector<TopMETObject> >();
}


TopMETObjectProducer::~TopMETObjectProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopMETObjectProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   Handle<vector<metType> > METs;
   iEvent.getByLabel(METLabel_, METs );
   
   // define vector of selected TopJet objects
   vector<TopMETObject> * ttMETs = new vector<TopMETObject>(); 
   for(size_t j=0; j<METs->size(); j++){
     if( (*METs)[j].et()>METcut_ ){
       
       TopMET amet((*METs)[j]);
       // add jet resolution info if demanded
       if(addResolutions_){
         //still to implement, for the moment some realistic average value was taken
	 amet.setResET(25.);
	 amet.setResEta(100000.);
	 amet.setResD(100000.);
	 amet.setResTheta(10000.);
	 amet.setResPhi(0.6);
	 amet.setResPinv(0.008);
      }
       ttMETs->push_back(TopMETObject(amet));
     }
   }

   // sort MET in ET
   std::sort(ttMETs->begin(),ttMETs->end(),eTComparator);

   // put genEvt object in Event
   auto_ptr<vector<TopMETObject> > myTopMETObjectProducer(ttMETs);
   iEvent.put(myTopMETObjectProducer);

}
