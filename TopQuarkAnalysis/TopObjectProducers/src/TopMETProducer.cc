#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMETProducer.h"

//
// constructors and destructor
//
TopMETProducer::TopMETProducer(const edm::ParameterSet & iConfig)
{
   // initialize the configurables
   metSrc_   	   = iConfig.getParameter<edm::InputTag>("metSource");
   metCut_         = iConfig.getParameter<double>       ("metCut");
   addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");
   addMuonCorr_    = iConfig.getParameter<bool>         ("addMuonCorrections");
   metResoFile_    = iConfig.getParameter<std::string>  ("metResoFile");
   muonSrc_        = iConfig.getParameter<edm::InputTag>("muonSource");   
   
   // construct resolution calculator
   if(addResolutions_){
     metResoCalc_ = new TopObjectResolutionCalc(metResoFile_);
   }
   
   // produces vector of mets
   produces<std::vector<TopMET> >("mets");
}


TopMETProducer::~TopMETProducer()
{
   if (addResolutions_) delete metResoCalc_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TopMETProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{     
  
   // Get the vector of MET's from the event
   edm::Handle<std::vector<METType> > mets;
   iEvent.getByLabel(metSrc_, mets);

   // read in the muons if demanded
   edm::Handle<std::vector<MuonType> > muons;
   if (addMuonCorr_) {
     iEvent.getByLabel(muonSrc_, muons);
   }
   
   // loop over mets
   std::vector<TopMET> * ttMETs = new std::vector<TopMET>(); 
   for (size_t j = 0; j < mets->size(); j++) {
     // apply cut ... FIXME: use PhysicsTools
     if((*mets)[j].et() > metCut_ ){
       // construct the TopMET
       TopMET amet((*mets)[j]);
       // add MET resolution info if demanded
       if (addResolutions_) {
         (*metResoCalc_)(amet);
       }
       // correct for muons if demanded
       if (addMuonCorr_) {
         for (size_t m=0; m<muons->size(); m++) {
           amet.setP4(reco::Particle::LorentzVector(
               amet.px()-(*muons)[m].px(),
               amet.py()-(*muons)[m].py(),
               0,
               sqrt(pow(amet.px()-(*muons)[m].px(), 2)+pow(amet.py()-(*muons)[m].py(), 2))
           ));
         }
       }
       // add the MET to the vector of TopMETs
       ttMETs->push_back(TopMET(amet));
     }
   }
   // sort MET in ET
   std::sort(ttMETs->begin(), ttMETs->end(), eTComparator_);

   // put genEvt object in Event
   auto_ptr<std::vector<TopMET> > myTopMETProducer(ttMETs);
   iEvent.put(myTopMETProducer, "mets");

}
