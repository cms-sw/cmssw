#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"


//
// constructors and destructor
//
TopElectronProducer::TopElectronProducer(const edm::ParameterSet & iConfig)
{
   // initialize the configurables
   electronSrc_      = iConfig.getParameter<edm::InputTag>("electronSource");
   electronPTcut_    = iConfig.getParameter<double>       ("electronPTcut");
   electronEtacut_   = iConfig.getParameter<double>       ("electronEtacut");
   electronLRcut_    = iConfig.getParameter<double>       ("electronLRcut");
   doGenMatch_       = iConfig.getParameter<bool>         ("doGenMatch");
   addResolutions_   = iConfig.getParameter<bool>         ("addResolutions");
   addLRValues_      = iConfig.getParameter<bool>         ("addLRValues");
   genPartSrc_       = iConfig.getParameter<edm::InputTag>("genParticleSource");
   electronResoFile_ = iConfig.getParameter<string>       ("electronResoFile");
   electronLRFile_   = iConfig.getParameter<string>       ("electronLRFile");
   
   // construct resolution calculator
   if (addResolutions_) {
     theResoCalc_ = new TopObjectResolutionCalc(electronResoFile_);
   }

   // produces vector of electrons
   produces<std::vector<TopElectron > >("electrons");
}


TopElectronProducer::~TopElectronProducer()
{
   if (addResolutions_) delete theResoCalc_;
}


//
// member functions
//


// ------------ method called to produce the data  ------------
void
TopElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{     
  
   // Get the collection of electrons from the event
   edm::Handle<std::vector<ElectronType> > electrons; 
   iEvent.getByLabel(electronSrc_, electrons);

   // Get the vector of generated particles from the event if needed
   edm::Handle<reco::CandidateCollection> particles;
   if (doGenMatch_) {
     iEvent.getByLabel(genPartSrc_, particles);
   }

   // prepare LR calculation if required
   if (addLRValues_) {
     theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, electronLRFile_, "");
   }

   // loop over electrons
   std::vector<TopElectron> * topElectrons = new std::vector<TopElectron>(); 
   for (size_t e=0; e<electrons->size(); e++) {
     if( (*electrons)[e].pt()>electronPTcut_ && fabs((*electrons)[e].eta())<electronEtacut_ ){
       // construct the TopElectron
       TopElectron anElectron((*electrons)[e]);
       // match to generated final state electrons
       // FIXME: check charge!
       if (doGenMatch_) {
         // initialize best match as null
         reco::Particle::LorentzVector nullV(0,0,0,0);
         reco::Particle::Point nullP(0,0,0);
         reco::GenParticleCandidate bestGenElectron(char(1), nullV, nullP, 0, 0);
         float bestDR = 0;
         // find the closest generated electron
         for(reco::CandidateCollection::const_iterator itGenElectron = particles->begin(); itGenElectron != particles->end(); ++itGenElectron) {
           reco::Candidate * aTmpGenElectron = const_cast<reco::Candidate *>(&*itGenElectron);
           reco::GenParticleCandidate aGenElectron = *(dynamic_cast<reco::GenParticleCandidate *>(aTmpGenElectron));
           if (abs(aGenElectron.pdgId())==11 && aGenElectron.status()==1) {
             HepLorentzVector aGenElectronHLV(aGenElectron.px(), aGenElectron.py(), aGenElectron.pz(), aGenElectron.energy()),
                              aRecElectronHLV((*electrons)[e].px(), (*electrons)[e].py(), (*electrons)[e].pz(), (*electrons)[e].energy());
             float currDR = aGenElectronHLV.deltaR(aRecElectronHLV);
             if (bestDR == 0 || currDR < bestDR) {
               bestGenElectron = aGenElectron;
               bestDR = currDR;
             }
           }
         }
         anElectron.setGenLepton(bestGenElectron);
       }
       // add resolution info if demanded
       if (addResolutions_) {
         (*theResoCalc_)(anElectron);
       }
       // add top lepton id LR info if requested
       if (addLRValues_) {
         theLeptonLRCalc_->calcLikelihood(anElectron, iEvent);
       }
       // add the electron to the vector of TopElectrons
       topElectrons->push_back(TopElectron(anElectron));
     }
   }
   // sort electrons in pT
   std::sort(topElectrons->begin(),topElectrons->end(),pTElectronComparator);

   // put genEvt object in Event
   auto_ptr<std::vector<TopElectron> > pOutElectron(topElectrons);
   iEvent.put(pOutElectron, "electrons");

   // destroy the lepton LR calculator
   if (addLRValues_) delete theLeptonLRCalc_;

}
