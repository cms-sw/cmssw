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
   electronPTcut_    = iConfig.getParameter<double>       ("electronPTcut");
   electronEtacut_   = iConfig.getParameter<double>       ("electronEtacut");
   electronLRcut_    = iConfig.getParameter<double>       ("electronLRcut");
   doGenMatch_       = iConfig.getParameter<bool>         ("doGenMatch");
   addResolutions_   = iConfig.getParameter<bool>         ("addResolutions");
   addLRValues_      = iConfig.getParameter<bool>         ("addLRValues");
   genPartSrc_       = iConfig.getParameter<edm::InputTag>("genParticleSource");
   electronLRFile_   = iConfig.getParameter<string>       ("electronLRFile");
   electronResoFile_ = iConfig.getParameter<string>       ("electronResoFile");
   
   //construct resolution calculator
   if (addResolutions_) {
     theResCalc_ = new TopObjectResolutionCalc(electronResoFile_);
   }

   //produces vector of electrons
   produces<vector<TopElectron > >("electrons");
}


TopElectronProducer::~TopElectronProducer()
{
   if (addResolutions_) delete theResCalc_;
}


//
// member functions
//


// ------------ method called to produce the data  ------------
void
TopElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{     
  
   if (addLRValues_) {
     theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, electronLRFile_, "");
   }

   // Get the vector of generated particles from the event
   edm::Handle<reco::CandidateCollection> particles;
   iEvent.getByLabel(genPartSrc_, particles);

   // Get the collection of electrons from the event
   edm::Handle<vector<ElectronType> > electrons; 
   iEvent.getByLabel("pixelMatchGsfElectrons", electrons );

   //loop over electrons
   vector<TopElectron> * topElectrons = new vector<TopElectron>(); 
   for (size_t e=0; e<electrons->size(); e++) {
     if( (*electrons)[e].pt()>electronPTcut_ && fabs((*electrons)[e].eta())<electronEtacut_ ){

       TopElectron anElectron((*electrons)[e]);

       // get generated particles
       edm::Handle<reco::CandidateCollection> particles;
       iEvent.getByLabel(genPartSrc_, particles);
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

       // add resolution info if demanded
       if (addResolutions_) {
         (*theResCalc_)(anElectron);
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
