#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"


//
// constructors and destructor
//
TopMuonProducer::TopMuonProducer(const edm::ParameterSet & iConfig)
{
   // initialize the configurables
   muonSrc_        = iConfig.getParameter<edm::InputTag>("muonSource");
   muonPTcut_      = iConfig.getParameter<double>       ("muonPTcut");
   muonEtacut_     = iConfig.getParameter<double>       ("muonEtacut");
   muonLRcut_      = iConfig.getParameter<double>       ("muonLRcut");
   doGenMatch_     = iConfig.getParameter<bool>         ("doGenMatch");
   addResolutions_ = iConfig.getParameter<bool>         ("addResolutions");
   addLRValues_    = iConfig.getParameter<bool>         ("addLRValues");
   genPartSrc_     = iConfig.getParameter<edm::InputTag>("genParticleSource");
   muonResoFile_   = iConfig.getParameter<std::string>  ("muonResoFile");
   muonLRFile_     = iConfig.getParameter<std::string>  ("muonLRFile");

   // construct resolution calculator
   if(addResolutions_){
     theResoCalc_ = new TopObjectResolutionCalc(muonResoFile_);
   }
   
   //produces vector of muons
   produces<std::vector<TopMuon > >("muons");
}


TopMuonProducer::~TopMuonProducer()
{
   if (addResolutions_) delete theResoCalc_;
}


//
// member functions
//


// ------------ method called to produce the data  ------------
void
TopMuonProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{     
  
   // Get the collection of muons from the event
   edm::Handle<std::vector<MuonType> > muons;
   iEvent.getByLabel(muonSrc_, muons);

   // Get the vector of generated particles from the event if needed
   edm::Handle<reco::CandidateCollection> particles;
   if (doGenMatch_) {
     iEvent.getByLabel(genPartSrc_, particles);
   }

   // prepare LR calculation if required
   if (addLRValues_) {
     theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, "", muonLRFile_);
   }

   // loop over muons
   std::vector<TopMuon> * topMuons = new std::vector<TopMuon>(); 
   for (size_t m=0; m<muons->size(); m++) {
     if( (*muons)[m].pt()>muonPTcut_ && fabs((*muons)[m].eta())<muonEtacut_ ){
       // construct the TopMuon
       TopMuon aMuon((*muons)[m]);
       // match to generated final state muons
       // FIXME: check charge!
       if (doGenMatch_) {
         // initialize best match as null
         reco::Particle::LorentzVector nullV(0,0,0,0);
         reco::Particle::Point nullP(0,0,0);
         reco::GenParticleCandidate bestGenMuon(char(1), nullV, nullP, 0, 0);
         float bestDR = 0;
         // find the closest generated muon
         for(reco::CandidateCollection::const_iterator itGenMuon = particles->begin(); itGenMuon != particles->end(); ++itGenMuon) {
           reco::Candidate * aTmpGenMuon = const_cast<reco::Candidate *>(&*itGenMuon);
           reco::GenParticleCandidate aGenMuon = *(dynamic_cast<reco::GenParticleCandidate *>(aTmpGenMuon));
           if (abs(aGenMuon.pdgId())==13 && aGenMuon.status()==1) {
             HepLorentzVector aGenMuonHLV(aGenMuon.px(), aGenMuon.py(), aGenMuon.pz(), aGenMuon.energy()),
                              aRecMuonHLV((*muons)[m].px(), (*muons)[m].py(), (*muons)[m].pz(), (*muons)[m].energy());
             float currDR = aGenMuonHLV.deltaR(aRecMuonHLV);
             if (bestDR == 0 || currDR < bestDR) {
               bestGenMuon = aGenMuon;
               bestDR = currDR;
             }
           }
         }
         aMuon.setGenLepton(bestGenMuon);
       }
       // add resolution info if demanded
       if (addResolutions_) {
         (*theResoCalc_)(aMuon);
       }
       // add top lepton id LR info if requested
       if (addLRValues_) {
         theLeptonLRCalc_->calcLikelihood(aMuon, iEvent);
       }
       // add the muon to the vector of TopMuons
       topMuons->push_back(TopMuon(aMuon));
     }
   }
   // sort muons in pT
   std::sort(topMuons->begin(), topMuons->end(), pTMuonComparator_);

   // put genEvt object in Event
   auto_ptr<std::vector<TopMuon> > pOutMuon(topMuons);
   iEvent.put(pOutMuon, "muons");
 
   // destroy the lepton LR calculator
   if (addLRValues_) delete theLeptonLRCalc_;

}
