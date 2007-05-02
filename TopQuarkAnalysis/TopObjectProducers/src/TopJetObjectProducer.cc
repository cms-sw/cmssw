#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetObjectProducer.h"

//
// constructors and destructor
//
TopJetObjectProducer::TopJetObjectProducer(const edm::ParameterSet& iConfig)
{
   jetTagsLabel_    = iConfig.getParameter< string > ("jetTagInput");
   lcaliJetsLabel_  = iConfig.getParameter< string > ("lcaliJetInput");
   recJetsLabel_    = iConfig.getParameter< string > ("recJetInput");
   bcaliJetsLabel_  = iConfig.getParameter< string > ("bcaliJetInput");
   recJetETcut_     = iConfig.getParameter< double > ("recJetETcut");
   calJetETcut_     = iConfig.getParameter< double > ("calJetETcut");
   jetEtaCut_       = iConfig.getParameter< double > ("jetEtacut");
   minNrConstis_    = iConfig.getParameter< int    > ("minNrConstis");
   addResolutions_  = iConfig.getParameter< bool   > ("addResolutions");

   produces<vector<TopJetObject> >();
}


TopJetObjectProducer::~TopJetObjectProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopJetObjectProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   Handle<vector<jetType> > recjets;
   iEvent.getByLabel(recJetsLabel_, recjets );
   Handle<vector<jetType> > lcalijets;
   iEvent.getByLabel(lcaliJetsLabel_, lcalijets );
   Handle<vector<jetType> > bcalijets;
   iEvent.getByLabel(bcaliJetsLabel_, bcalijets );
   Handle<vector<JetTag> > jetTags;
   iEvent.getByLabel(jetTagsLabel_, jetTags );
   
   
   
   
   // define vector of selected TopJetProducer objects
   vector<TopJetObject> * ttJets = new vector<TopJetObject>(); 
   for(size_t j=0; j<recjets->size(); j++){
     if( (*recjets)[j].et()>recJetETcut_ && fabs((*recjets)[j].eta())<jetEtaCut_ && (*recjets)[j].nConstituents()>minNrConstis_){
       
       TopJetObject jetObj;
       jetObj.setRecJet((*recjets)[j]);
       
       // loop over cal jets to find corresponding jet
       TopJet lcaljet, bcaljet;
       bool lcjFound = false;
       bool bcjFound = false;
       for(size_t lcj=0; lcj<lcalijets->size(); lcj++){
         if(ROOT::Math::VectorUtil::DeltaR((*recjets)[j].p4(),(*lcalijets)[lcj].p4()) < 0.01) {
	   lcjFound = true;
	   TopJet ajet((*lcalijets)[lcj]);
           if(addResolutions_){
             //still to implement, for the moment some realistic average value was taken
	     ajet.setResET(12.);
	     ajet.setResEta(0.03);
	     ajet.setResPhi(0.03);
	     ajet.setResTheta(0.03);
	     ajet.setResD(0.12);
	     ajet.setResPinv(0.001);
	   }
	   jetObj.setLCalJet(ajet);
	 }
       }
       for(size_t bcj=0; bcj<bcalijets->size(); bcj++){
         if(ROOT::Math::VectorUtil::DeltaR((*recjets)[j].p4(),(*bcalijets)[bcj].p4()) < 0.01) {
	   bcjFound = true;
	   TopJet ajet((*bcalijets)[bcj]);
           if(addResolutions_){
             //still to implement, for the moment some realistic average value was taken
	     ajet.setResET(12.);
	     ajet.setResEta(0.03);
	     ajet.setResPhi(0.03);
	     ajet.setResTheta(0.03);
	     ajet.setResD(0.12);
	     ajet.setResPinv(0.001);
	   }
	   jetObj.setBCalJet(ajet);
	 }
       }
       
       // if cal jets found, add b-tag info if available
       if (lcjFound && bcjFound){
         if(jetObj.getLCalJet().et()>calJetETcut_ && jetObj.getBCalJet().et()>recJetETcut_){
           for(size_t t=0; t<jetTags->size(); t++){
             if(ROOT::Math::VectorUtil::DeltaR((*recjets)[j].p4(),(*jetTags)[t].jet().p4()) < 0.0001){
	       jetObj.setBdiscriminant((*jetTags)[t].discriminator());
	     }
	   }
	 }
       }
       else 
       { 
         cout<<"no cal jet found "<<endl;
       }
       ttJets->push_back(jetObj);
     }
   }

   // sort jets in ET
   std::sort(ttJets->begin(),ttJets->end(),eTComparator);

   // put genEvt object in Event
   auto_ptr<vector<TopJetObject> > myTopJetObjectProducer(ttJets);
   iEvent.put(myTopJetObjectProducer);
   

}
