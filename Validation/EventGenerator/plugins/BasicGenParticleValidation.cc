/*class BasicGenParticleValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
 
#include "Validation/EventGenerator/interface/BasicGenParticleValidation.h"

#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"

using namespace edm;

BasicGenParticleValidation::BasicGenParticleValidation(const edm::ParameterSet& iPSet): 
  wmanager_(iPSet,consumesCollector()),
  hepmcCollection_(iPSet.getParameter<edm::InputTag>("hepmcCollection")),
  genparticleCollection_(iPSet.getParameter<edm::InputTag>("genparticleCollection")),
  genjetCollection_(iPSet.getParameter<edm::InputTag>("genjetsCollection")),
  matchPr_(iPSet.getParameter<double>("matchingPrecision")),
  verbosity_(iPSet.getUntrackedParameter<unsigned int>("verbosity",0))
{    

  hepmcCollectionToken_=consumes<HepMCProduct>(hepmcCollection_);
  genparticleCollectionToken_=consumes<reco::GenParticleCollection>(genparticleCollection_);
  genjetCollectionToken_=consumes<reco::GenJetCollection>(genjetCollection_);

}

BasicGenParticleValidation::~BasicGenParticleValidation() {}



void BasicGenParticleValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &){
  ///Setting the DQM top directories
  DQMHelper dqm(&i); i.setCurrentFolder("Generator/GenParticles");
  ///Booking the ME's
  
  // Number of analyzed events
  nEvt = dqm.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1.,"","Number of Events");
  
  ///multiplicity
  genPMultiplicity = dqm.book1dHisto("genPMultiplicty", "Log(No. all GenParticles)", 50, -1, 5,"log_{10}(N_{All GenParticles})","Number of Events"); //Log
  //difference in HepMC and reco multiplicity
  genMatched = dqm.book1dHisto("genMatched", "Difference reco - matched", 50, -25, 25,"N_{All GenParticles}-N_{Matched}","Number of Events");
  //multiple matching
  multipleMatching = dqm.book1dHisto("multipleMatching", "multiple reco HepMC matching", 50, 0, 50,"N_{multiple reco HepMC matching}","Number of Events");
  //momentum difference of matched particles
  matchedResolution = dqm.book1dHisto("matchedResolution", "log10(momentum difference of matched particles)", 70, -10., -3.,"log_{10}(#DeltaP_{matched Particles})","Number of Events");
  
  // GenJet general distributions
  genJetMult = dqm.book1dHisto("genJetMult", "GenJet multiplicity", 50, 0, 50,"N_{gen-jets}","Number of Events");
  genJetEnergy = dqm.book1dHisto("genJetEnergy", "Log10(GenJet energy)", 60, -1, 5,"log_{10}(E^{gen-jets}) (log_{10}(GeV))","Number of Events");
  genJetPt = dqm.book1dHisto("genJetPt", "Log10(GenJet pt)", 60, -1, 5,"log_{10}(P_{t}^{gen-jets}) (log_{10}(GeV))","Number of Events");
  genJetEta = dqm.book1dHisto("genJetEta", "GenJet eta", 220, -11, 11,"#eta^{gen-jets}","Number of Events");
  genJetPhi = dqm.book1dHisto("genJetPhi", "GenJet phi", 360, -180, 180,"#phi^{gen-jets} (rad)","Number of Events");
  genJetDeltaEtaMin = dqm.book1dHisto("genJetDeltaEtaMin", "GenJet minimum rapidity gap", 30, 0, 30,"#delta#eta_{min}^{gen-jets}","Number of Events");
  
  genJetPto1 = dqm.book1dHisto("genJetPto1", "GenJet multiplicity above 1 GeV", 50, 0, 50,"N_{gen-jets P_{t}>1GeV}","Number of Events");
  genJetPto10 = dqm.book1dHisto("genJetPto10", "GenJet multiplicity above 10 GeV", 50, 0, 50,"N_{gen-jets P_{t}>10GeV}","Number of Events");
  genJetPto100 = dqm.book1dHisto("genJetPto100", "GenJet multiplicity above 100 GeV", 50, 0, 50,"N_{gen-jets P_{t}>100GeV}","Number of Events");
  genJetCentral = dqm.book1dHisto("genJetCentral", "GenJet multiplicity |eta|.lt.2.5", 50, 0, 50,"N_{gen-jets |#eta|#leq2.5}","Number of Events");
  
  genJetTotPt = dqm.book1dHisto("genJetTotPt", "Log10(GenJet total pt)", 100, -5, 5,"log_{10}(#SigmaP_{t}^{gen-jets}) (log_{10}(GeV))","Number of Events");
  
  return;
}

void BasicGenParticleValidation::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 

  unsigned int initSize = 1000;

  ///Gathering the HepMCProduct information
  edm::Handle<HepMCProduct> evt;
  iEvent.getByToken(hepmcCollectionToken_, evt);

  //Get HepMC EVENT
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));

  double weight =    wmanager_.weight(iEvent);

  nEvt->Fill(0.5, weight);

  std::vector<const HepMC::GenParticle*> hepmcGPCollection;
  std::vector<int> barcodeList;
  hepmcGPCollection.reserve(initSize);
  barcodeList.reserve(initSize);

  //Looping through HepMC::GenParticle collection to search for status 1 particles
  for (HepMC::GenEvent::particle_const_iterator iter = myGenEvent->particles_begin(); iter != myGenEvent->particles_end(); ++iter){
    if ( (*iter)->status() == 1) {
      hepmcGPCollection.push_back(*iter);
      barcodeList.push_back((*iter)->barcode());
      if ( verbosity_ > 0 ) {
        std::cout << "HepMC " << std::setw(14) << std::fixed << (*iter)->pdg_id() << std::setw(14) << std::fixed << (*iter)->momentum().px() << std::setw(14) << std::fixed 
                  << (*iter)->momentum().py() << std::setw(14) << std::fixed << (*iter)->momentum().pz() << std::endl;
      }
    }
  }
  

  // Gather information on the reco::GenParticle collection
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genparticleCollectionToken_, genParticles );
  
  std::vector<const reco::GenParticle*> particles;
  particles.reserve(initSize);
  for (reco::GenParticleCollection::const_iterator iter=genParticles->begin();iter!=genParticles->end();++iter){
    if ( (*iter).status() == 1) { 
      particles.push_back(&*iter); 
      if ( verbosity_ > 0 ) {
        std::cout << "reco  " << std::setw(14) << std::fixed << (*iter).pdgId() << std::setw(14) << std::fixed << (*iter).px() 
                  << std::setw(14) << std::fixed << (*iter).py() << std::setw(14) << std::fixed << (*iter).pz() << std::endl;
      }
    }
  }

  unsigned int nReco = particles.size();
  unsigned int nHepMC = hepmcGPCollection.size();

  genPMultiplicity->Fill(std::log10(nReco), weight);

  // Define vector containing index of hepmc corresponding to the reco::GenParticle
  std::vector<int> hepmcMatchIndex;
  hepmcMatchIndex.reserve(initSize);

  // Matching procedure

  // Check array size consistency

  if ( nReco != nHepMC ) {
    edm::LogWarning("CollectionSizeInconsistency") << "Collection size inconsistency: HepMC::GenParticle = " << nHepMC << " reco::GenParticle = " << nReco;
  }

  // Match each HepMC with a reco

  for ( unsigned int i = 0; i < nReco; ++i ){
    for ( unsigned int j = 0; j < nHepMC; ++j ){
      if ( matchParticles( hepmcGPCollection[j], particles[i] ) ) { 
        hepmcMatchIndex.push_back((int)j); 
        if ( hepmcGPCollection[j]->momentum().rho() != 0. ) { 
          double reso = 1.-particles[i]->p()/hepmcGPCollection[j]->momentum().rho();
          if ( verbosity_ > 0 ) { 
            std::cout << "Matching momentum: reco = " << particles[i]->p() << " HepMC = " 
                      << hepmcGPCollection[j]->momentum().rho() << " resoultion = " << reso << std::endl;
          }
          matchedResolution->Fill(std::log10(std::fabs(reso)),weight); }
        continue; 
      }
    }
  }

  // Check unicity of matching

  unsigned int nMatched = hepmcMatchIndex.size();
  
  if ( nMatched != nReco ) {
    edm::LogWarning("IncorrectMatching") << "Incorrect number of matched indexes: GenParticle = " << nReco << " matched indexes = " << nMatched;
  }
  genMatched->Fill(int(nReco-nMatched),weight);

  unsigned int nWrMatch = 0;

  for ( unsigned int i = 0; i < nMatched; ++i ){
    for (unsigned int j = i+1; j < nMatched; ++j ){
      if ( hepmcMatchIndex[i] == hepmcMatchIndex[j] ) {
        int theIndex = hepmcMatchIndex[i];
        edm::LogWarning("DuplicatedMatching") << "Multiple matching occurencies for GenParticle barcode = " << barcodeList[theIndex];
        nWrMatch++;
      }
    }
  }
  multipleMatching->Fill(int(nWrMatch),weight);

  // Gather information in the GenJet collection
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken(genjetCollectionToken_, genJets );

  int nJets = 0;
  int nJetso1 = 0;
  int nJetso10 = 0;
  int nJetso100 = 0;
  int nJetsCentral = 0;
  double totPt = 0.;
  
  std::vector<double> jetEta;
  jetEta.reserve(initSize);

  for (reco::GenJetCollection::const_iterator iter=genJets->begin();iter!=genJets->end();++iter){
    nJets++;
    double pt = (*iter).pt();
    totPt += pt;
    if (pt > 1.) nJetso1++;
    if (pt > 10.) nJetso10++;
    if (pt > 100.) nJetso100++;
    double eta = (*iter).eta();
    if ( std::fabs(eta) < 2.5 ) nJetsCentral++;
    jetEta.push_back(eta);

    genJetEnergy->Fill(std::log10((*iter).energy()),weight);
    genJetPt->Fill(std::log10(pt),weight);
    genJetEta->Fill(eta,weight);
    genJetPhi->Fill((*iter).phi()/CLHEP::degree,weight);
  }

  genJetMult->Fill(nJets,weight);
  genJetPto1->Fill(nJetso1,weight);
  genJetPto10->Fill(nJetso10,weight);
  genJetPto100->Fill(nJetso100,weight);
  genJetCentral->Fill(nJetsCentral,weight);

  genJetTotPt->Fill(std::log10(totPt),weight);

  double deltaEta = 999.;
  if ( jetEta.size() > 1 ) {
    for (unsigned int i = 0; i < jetEta.size(); i++){
      for (unsigned int j = i+1; j < jetEta.size(); j++){
        deltaEta = std::min(deltaEta,std::fabs(jetEta[i]-jetEta[j]));
      }
    }
  }

  genJetDeltaEtaMin->Fill(deltaEta,weight);

  delete myGenEvent;
}//analyze

bool BasicGenParticleValidation::matchParticles(const HepMC::GenParticle*& hepmcP, const reco::GenParticle*& recoP){

  bool state = false;

  if ( hepmcP->pdg_id() != recoP->pdgId() ) return state;
  if ( std::fabs(hepmcP->momentum().px()-recoP->px()) < std::fabs(matchPr_*hepmcP->momentum().px()) && 
       std::fabs(hepmcP->momentum().py()-recoP->py()) < std::fabs(matchPr_*hepmcP->momentum().py()) && 
       std::fabs(hepmcP->momentum().pz()-recoP->pz()) < std::fabs(matchPr_*hepmcP->momentum().pz())) {
    state = true; }

  return state;

}
