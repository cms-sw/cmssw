#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"

//
// constructors and destructor
//
TopJetProducer::TopJetProducer(const edm::ParameterSet& iConfig)
{
   jetTagsLabel_    	= iConfig.getParameter< string > ("jetTagInput");
   recJetsLabel_    	= iConfig.getParameter< string > ("recJetInput");
   caliJetsLabel_  	= iConfig.getParameter< string > ("caliJetInput");
   recJetETcut_     	= iConfig.getParameter< double > ("recJetETcut");
   jetEtaCut_       	= iConfig.getParameter< double > ("jetEtacut");
   minNrConstis_    	= iConfig.getParameter< int    > ("minNrConstis");
   addResolutions_  	= iConfig.getParameter< bool   > ("addResolutions");
   caliJetResoFile_ 	= iConfig.getParameter< string > ("caliJetResoFile");
   
   //construct resolution calculator
   if(addResolutions_) jetsResCalc  = new TopObjectResolutionCalc(caliJetResoFile_);

   produces<vector<TopJet> >();
}


TopJetProducer::~TopJetProducer()
{
   if(addResolutions_) delete jetsResCalc;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{     
  
   // Get the vector of generated particles from the event
   edm::Handle<vector<JetType> > recjets;
   iEvent.getByLabel(recJetsLabel_, recjets );
   edm::Handle<vector<JetType> > calijets;
   iEvent.getByLabel(caliJetsLabel_, calijets );
   edm::Handle<vector<reco::JetTag> > jetTags;
   iEvent.getByLabel(jetTagsLabel_, jetTags );
   
   
   
   
   // define vector of selected TopJetProducer objects
   vector<TopJet> * ttJets = new vector<TopJet>(); 
   for(size_t j=0; j<recjets->size(); j++){
     if( (*recjets)[j].et()>recJetETcut_ && fabs((*recjets)[j].eta())<jetEtaCut_ && (*recjets)[j].nConstituents()>minNrConstis_){
       
       // loop over cal jets to find corresponding jet
       TopJet ajet;
       bool cjFound = false;
       for(size_t cj=0; cj<calijets->size(); cj++){
         if(ROOT::Math::VectorUtil::DeltaR((*recjets)[j].p4(),(*calijets)[cj].p4()) < 0.01) {
	   cjFound = true;
	   ajet = TopJet((*calijets)[cj]);
           ajet.setRecJet((*recjets)[j]);
           if(addResolutions_){
             (*jetsResCalc)(ajet);
	   }
	 }
       }

       // if cal jet found, add b-tag info if available
       if (cjFound){
         for(size_t t=0; t<jetTags->size(); t++){
           if(ROOT::Math::VectorUtil::DeltaR((*recjets)[j].p4(),(*jetTags)[t].jet().p4()) < 0.0001){
	     ajet.setBdiscriminant((*jetTags)[t].discriminator());
	   }
	 }
       }
       else 
       { 
         cout<<"no cal jet found "<<endl;
       }
       ttJets->push_back(ajet);
     }
   }

   // sort jets in ET
   std::sort(ttJets->begin(),ttJets->end(),eTComparator);

   // put genEvt  in Event
   auto_ptr<vector<TopJet> > myTopJetProducer(ttJets);
   iEvent.put(myTopJetProducer);
   

}
