#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETProducer.h"

//
// constructors and destructor
//
TopMETProducer::TopMETProducer(const edm::ParameterSet& iConfig)
{
   METLabel_   	   = iConfig.getParameter< string > ("METInput");
   METcut_         = iConfig.getParameter< double > ("METcut");
   addResolutions_ = iConfig.getParameter< bool   > ("addResolutions");
   metResoFile_    = iConfig.getParameter< string > ("metResoFile");
   
   //construct resolution calculator
   if(addResolutions_){
     metResCalc = new TopObjectResolutionCalc(metResoFile_);
   }
   
   //produces vector of mets
   produces<vector<TopMET> >();
}


TopMETProducer::~TopMETProducer()
{
   if(addResolutions_) delete metResCalc;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopMETProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   edm::Handle<vector<METType> > METs;
   iEvent.getByLabel(METLabel_, METs );
   
   // define vector of selected TopJet objects
   vector<TopMET> * ttMETs = new vector<TopMET>(); 
   for(size_t j=0; j<METs->size(); j++){
     if( (*METs)[j].et()>METcut_ ){
       
       TopMET amet((*METs)[j]);
       // add jet resolution info if demanded
       if(addResolutions_){
         (*metResCalc)(amet);
       }
       ttMETs->push_back(TopMET(amet));
     }
   }

   // sort MET in ET
   std::sort(ttMETs->begin(),ttMETs->end(),eTComparator);

   // put genEvt object in Event
   auto_ptr<vector<TopMET> > myTopMETProducer(ttMETs);
   iEvent.put(myTopMETProducer);

}
