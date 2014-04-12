#include "BoostedTopProducer.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"

#include <string>
#include <sstream>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BoostedTopProducer::BoostedTopProducer(const edm::ParameterSet& iConfig) :
  eleToken_   (consumes<std::vector<pat::Electron> >(iConfig.getParameter<edm::InputTag>  ("electronLabel"))),
  muoToken_   (consumes<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>  ("muonLabel"))),
  jetToken_   (consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>  ("jetLabel"))),
  metToken_   (consumes<std::vector<pat::MET> >(iConfig.getParameter<edm::InputTag>  ("metLabel"))),
  solToken_   (mayConsume<TtSemiLeptonicEvent>(iConfig.getParameter<edm::InputTag>  ("solLabel"))),
  caloIsoCut_ (iConfig.getParameter<double>         ("caloIsoCut") ),
  mTop_       (iConfig.getParameter<double>         ("mTop") )
{
  //register products
  produces<std::vector<reco::CompositeCandidate> > ();
}


BoostedTopProducer::~BoostedTopProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
BoostedTopProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;

  bool debug = false;

  // -----------------------------------------------------
  // get the bare PAT objects
  // -----------------------------------------------------
   edm::Handle<std::vector<pat::Muon> > muonHandle;
   iEvent.getByToken(muoToken_,muonHandle);
   std::vector<pat::Muon> const & muons = *muonHandle;

   edm::Handle<std::vector<pat::Jet> > jetHandle;
   iEvent.getByToken(jetToken_,jetHandle);
   std::vector<pat::Jet> const & jets = *jetHandle;

   edm::Handle<std::vector<pat::Electron> > electronHandle;
   iEvent.getByToken(eleToken_,electronHandle);
   std::vector<pat::Electron> const & electrons = *electronHandle;

   edm::Handle<std::vector<pat::MET> > metHandle;
   iEvent.getByToken(metToken_,metHandle);
   std::vector<pat::MET> const & mets = *metHandle;

   // -----------------------------------------------------
   // Event Preselection:
   //    <= 1 isolated electron or muon
   //    >= 1 electron or muon
   //    >= 2 jets
   //    >= 1 missing et
   //
   // To explain:
   //    We want to look at leptons within "top jets" in some
   //    cases. This means the isolation will kill those events.
   //    However, if there IS an isolated lepton, we want only
   //    one of them.
   //
   //    So to select the prompt W lepton, the logic is:
   //    1. If there is an isolated lepton, accept it as the W lepton.
   //    2. Else, take the highest Pt lepton (possibly non-isolated)
   //
   // -----------------------------------------------------
   bool preselection = true;

   // This will hold the prompt W lepton candidate, and a
   // maximum pt decision variable
   double maxWLeptonPt = -1;
   //reco::Candidate const * Wlepton = 0;

   // ----------------------
   // Find isolated muons, and highest pt lepton
   // ----------------------
   std::vector<pat::Muon>::const_iterator isolatedMuon     = muons.end();
   std::vector<pat::Muon>::const_iterator muon = muons.end();
   bool nIsolatedMuons = 0;
   std::vector<pat::Muon>::const_iterator muonIt = muons.begin(),
     muonEnd = muons.end();
   for (; muonIt != muonEnd; ++muonIt ) {

     // Find highest pt lepton
     double pt = muonIt->pt();
     if ( pt > maxWLeptonPt ) {
       maxWLeptonPt = pt;
       muon = muonIt;
     }

     // Find any isolated muons
     double caloIso = muonIt->caloIso();
     if ( caloIso >= 0 && caloIso < caloIsoCut_ ) {
       nIsolatedMuons++;
       isolatedMuon = muonIt;
     }
   }

   // ----------------------
   // Find isolated electrons, and highest pt lepton
   // ----------------------
   std::vector<pat::Electron>::const_iterator isolatedElectron     = electrons.end();
   std::vector<pat::Electron>::const_iterator electron = electrons.end();
   bool nIsolatedElectrons = 0;
   std::vector<pat::Electron>::const_iterator electronIt = electrons.begin(),
     electronEnd = electrons.end();
   for (; electronIt != electronEnd; ++electronIt ) {

     // Find highest pt lepton
     double pt = electronIt->pt();
     if ( pt > maxWLeptonPt ) {
       maxWLeptonPt = pt;
       electron = electronIt;
     }

     // Find any isolated electrons
     double caloIso = electronIt->caloIso();
     if ( caloIso >= 0 && caloIso < caloIsoCut_ ) {
       nIsolatedElectrons++;
       isolatedElectron = electronIt;
     }
   }


   // ----------------------
   // Now decide on the "prompt" lepton from the W:
   // Choose isolated leptons over all, and if no isolated,
   // then take highest pt lepton.
   // ----------------------
   bool isMuon = true;
   if      ( isolatedMuon     != muonEnd     ) { muon     = isolatedMuon;     isMuon = true; }
   else if ( isolatedElectron != electronEnd ) { electron = isolatedElectron; isMuon = false; }
   else {
     // Set to the highest pt lepton
     if      ( muon != muonEnd && electron == electronEnd ) isMuon = true;
     else if ( muon == muonEnd && electron != electronEnd ) isMuon = false;
     else if ( muon != muonEnd && electron != electronEnd ) {
       isMuon =  muon->pt() > electron->pt();
     }
   }

   // ----------------------
   // Veto events that have more than one isolated lepton
   // ----------------------
   int nIsolatedLeptons = nIsolatedMuons + nIsolatedElectrons;
   if ( nIsolatedLeptons > 1 ) {
     preselection = false;
   }

   // ----------------------
   // Veto events that have no prompt lepton candidates
   // ----------------------
   if ( muon == muonEnd && electron == electronEnd ) {
     preselection = false;
   }

   // ----------------------
   // Veto events with < 2 jets or no missing et
   // ----------------------
   if ( jets.size() < 2 ||
	mets.size() == 0 ) {
     preselection = false;
   }

   bool write = false;



   // -----------------------------------------------------
   //
   // CompositeCandidates to store the event solution.
   // This will take one of two forms:
   //    a) lv jj jj   Full reconstruction.
   //
   //   ttbar->
   //       (hadt -> (hadW -> hadp + hadq) + hadb) +
   //       (lept -> (lepW -> lepton + neutrino) + lepb)
   //
   //    b) lv jj (j)  Partial reconstruction, associate
   //                  at least 1 jet to the lepton
   //                  hemisphere, and at least one jet in
   //                  the opposite hemisphere.
   //
   //    ttbar->
   //        (hadt -> (hadJet1 [+ hadJet2] ) ) +
   //        (lept -> (lepW -> lepton + neutrino) + lepJet1 )
   //
   // There will also be two subcategories of (b) that
   // will correspond to physics cases:
   //
   //    b1)           Lepton is isolated: Moderate ttbar mass.
   //    b2)           Lepton is nonisolated: High ttbar mass.
   //
   // -----------------------------------------------------
   reco::CompositeCandidate ttbar("ttbar");
   AddFourMomenta addFourMomenta;


   // Main decisions after preselection
   if ( preselection ) {

     if ( debug ) cout << "Preselection is satisfied" << endl;

     if ( debug ) cout << "Jets.size() = " << jets.size() << endl;

     // This will be modified for the z solution, so make a copy
     pat::MET              neutrino( mets[0] );


     // 1. First examine the low mass case with 4 jets and widely separated
     //    products. We take out the TtSemiLeptonicEvent from the TQAF and
     //    form the ttbar invariant mass.
     if ( jets.size() >= 4 ) {

       if ( debug ) cout << "Getting ttbar semileptonic solution" << endl;

       // get the ttbar semileptonic event solution if there are more than 3 jets
       edm::Handle< TtSemiLeptonicEvent > eSol;
       iEvent.getByToken(solToken_, eSol);

       // Have solution, continue
       if ( eSol.isValid() ) {
	 if ( debug ) cout << "Got a nonzero size solution vector" << endl;
	 // Just set the ttbar solution to the best ttbar solution from
	 // TtSemiEvtSolutionMaker
	 ttbar = eSol->eventHypo(TtSemiLeptonicEvent::kMVADisc);
	 write = true;
       }
       // No ttbar solution with 4 jets, something is weird, print a warning
       else {
	 edm::LogWarning("DataNotFound") << "BoostedTopProducer: Cannot find TtSemiEvtSolution\n";
       }
     }
     // 2. With 2 or 3 jets, we decide based on the separation between
     // the lepton and the closest jet in that hemisphere whether to
     // consider it "moderate" or "high" mass.
     else if ( jets.size() == 2 || jets.size() == 3 ) {

       // ------------------------------------------------------------------
       // First create a leptonic W candidate
       // ------------------------------------------------------------------
       reco::CompositeCandidate lepW("lepW");

       if ( isMuon ) {
	 if ( debug ) cout << "Adding muon as daughter" << endl;
	 lepW.addDaughter(  *muon,     "muon" );
       } else {
	 if ( debug ) cout << "Adding electron as daughter" << endl;
	 lepW.addDaughter( *electron, "electron" );
       }
       if ( debug ) cout << "Adding neutrino as daughter" << endl;
       lepW.addDaughter  ( neutrino, "neutrino");
       addFourMomenta.set( lepW );

       //bool nuzHasComplex = false;
       METzCalculator zcalculator;

       zcalculator.SetMET( neutrino );
       if ( isMuon )
 	 zcalculator.SetMuon( *muon );
       else
	 zcalculator.SetMuon( *electron ); // This name is misleading, should be setLepton
       double neutrinoPz = zcalculator.Calculate(1);// closest to the lepton Pz
       //if (zcalculator.IsComplex()) nuzHasComplex = true;
       // Set the neutrino pz
       neutrino.setPz( neutrinoPz );

       if ( debug ) cout << "Set neutrino pz to " << neutrinoPz << endl;

       // ------------------------------------------------------------------
       // Next ensure that there is a jet within the hemisphere of the
       // leptonic W, and one in the opposite hemisphere
       // ------------------------------------------------------------------
       reco::CompositeCandidate hadt("hadt");
       reco::CompositeCandidate lept("lept");
       if ( debug ) cout << "Adding lepW as daughter" << endl;
       lept.addDaughter( lepW, "lepW" );

       std::string hadName("hadJet");
       std::string lepName("lepJet");

       // Get the W momentum
       TLorentzVector p4_W (lepW.px(), lepW.py(), lepW.pz(), lepW.energy() );

       // Loop over the jets
       std::vector<pat::Jet>::const_iterator jetit = jets.begin(),
	 jetend = jets.end();
       unsigned long ii = 1; // Count by 1 for naming histograms
       for ( ; jetit != jetend; ++jetit, ++ii ) {
	 // Get this jet's momentum
	 TLorentzVector p4_jet (jetit->px(), jetit->py(), jetit->pz(), jetit->energy() );

	 // Calculate psi (like DeltaR, only more invariant under Rapidity)
	 double psi = Psi( p4_W, p4_jet, mTop_ );

	 // Get jets that are in the leptonic hemisphere
	 if ( psi < TMath::Pi() ) {
	   // Add this jet to the leptonic top
	   std::stringstream s;
	   s << lepName << ii;
	   if ( debug ) cout << "Adding daughter " << s.str() << endl;
	   lept.addDaughter( *jetit, s.str() );
	 }
	 // Get jets that are in the hadronic hemisphere
	 if ( psi > TMath::Pi() ) {
	   // Add this jet to the hadronic top. We don't
	   // make any W hypotheses in this case, since
	   // we cannot determine which of the three
	   // jets are merged.
	   std::stringstream s;
	   s << hadName << ii;
	   if ( debug ) cout << "Adding daughter " << s.str() << endl;
	   hadt.addDaughter( *jetit, s.str() );

	 }
       } // end loop over jets

       addFourMomenta.set( lept );
       addFourMomenta.set( hadt );

       bool lepWHasJet = lept.numberOfDaughters() >= 2; // W and >= 1 jet
       bool hadWHasJet = hadt.numberOfDaughters() >= 1; // >= 1 jet
       if ( lepWHasJet && hadWHasJet ) {
	 if ( debug ) cout << "Adding daughters lept and hadt" << endl;
	 ttbar.addDaughter( lept, "lept");
	 ttbar.addDaughter( hadt, "hadt");
	 addFourMomenta.set( ttbar );
	 write = true;
       } // end of hadronic jet and leptonic jet


     } // end if there are 2 or 3 jets

   } // end if preselection is satisfied

   // Write the solution to the event record
   std::vector<reco::CompositeCandidate> ttbarList;
   if ( write ) {
     if ( debug ) cout << "Writing out" << endl;
     ttbarList.push_back( ttbar );
   }
   std::auto_ptr<std::vector<reco::CompositeCandidate> > pTtbar ( new std::vector<reco::CompositeCandidate>(ttbarList) );
   iEvent.put( pTtbar );


}

// ------------ method called once each job just before starting event loop  ------------
void
BoostedTopProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
BoostedTopProducer::endJob() {
}

double
BoostedTopProducer::Psi(const TLorentzVector& p1, const TLorentzVector& p2, double mass) {

	TLorentzVector ptot = p1 + p2;
	Double_t theta1 = TMath::ACos( (p1.Vect().Dot(ptot.Vect()))/(p1.P()*ptot.P()) );
	Double_t theta2 = TMath::ACos( (p2.Vect().Dot(ptot.Vect()))/(p2.P()*ptot.P()) );
	//Double_t sign = 1.;
	//if ( (theta1+theta2) > (TMath::Pi()/2) ) sign = -1.;
	double th1th2 = theta1 + theta2;
	double psi = (p1.P()+p2.P())*TMath::Abs(TMath::Sin(th1th2))/(2.* mass );
	if ( th1th2 > (TMath::Pi()/2) )
		psi = (p1.P()+p2.P())*( 1. + TMath::Abs(TMath::Cos(th1th2)))/(2.* mass );

	return psi;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BoostedTopProducer);
