#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetTagger.h"
#include "AnalysisDataFormats/TopObjects/interface/CATopJetTagInfo.h"

using namespace std;
using namespace reco;
using namespace edm;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

struct GreaterByPtCandPtr {
  bool operator()( const edm::Ptr<reco::Candidate> & t1, const edm::Ptr<reco::Candidate> & t2 ) const {
    return t1->pt() > t2->pt();
  }
};


//
// constructors and destructor
//
CATopJetTagger::CATopJetTagger(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<InputTag>("src") ),
  TopMass_(iConfig.getParameter<double>("TopMass") ),
  TopMassMin_(iConfig.getParameter<double>("TopMassMin") ),
  TopMassMax_(iConfig.getParameter<double>("TopMassMax") ),

  WMass_(iConfig.getParameter<double>("WMass") ),
  WMassMin_(iConfig.getParameter<double>("WMassMin") ),
  WMassMax_(iConfig.getParameter<double>("WMassMax") ),

  MinMassMin_(iConfig.getParameter<double>("MinMassMin") ),
  MinMassMax_(iConfig.getParameter<double>("MinMassMax") ),

  verbose_(iConfig.getParameter<bool>("verbose") )
{
  produces<CATopJetTagInfoCollection>();
}


CATopJetTagger::~CATopJetTagger()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CATopJetTagger::produce( edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Set up output list
  auto_ptr<CATopJetTagInfoCollection> tagInfos(new CATopJetTagInfoCollection() );

  // Here is the four-momentum adder
   AddFourMomenta addFourMomenta;

   // Get the input list of basic jets corresponding to the hard jets
   Handle<View<Jet> > pBasicJets;
   iEvent.getByLabel(src_, pBasicJets);

   // Get a convenient handle
   View<Jet> const & hardJets = *pBasicJets;
   
   // Now loop over the hard jets and do kinematic cuts
   View<Jet>::const_iterator ihardJet = hardJets.begin(),
     ihardJetEnd = hardJets.end();
   size_t iihardJet = 0;
   for ( ; ihardJet != ihardJetEnd; ++ihardJet, ++iihardJet ) {

     if ( verbose_ ) cout << "Processing ihardJet with pt = " << ihardJet->pt() << endl;

     // Get subjets
     Jet::Constituents subjets = ihardJet->getJetConstituents();

     // Initialize output variables
     // Get a ref to the hard jet
     RefToBase<Jet> ref( pBasicJets, iihardJet );
     // Get properties
     CATopJetProperties properties;
     properties.nSubJets = subjets.size();  // number of subjets
     properties.topMass = ihardJet->mass();      // jet mass
     properties.wMass = 99999.;                  // best W mass
     properties.minMass = 999999.;            // minimum mass pairing

     // Require at least three subjets in all cases, if not, untagged
     if ( verbose_ ) cout << "nSubJets = " << properties.nSubJets << endl;
     if ( properties.nSubJets >= 3 ) {

       // Take the highest 3 pt subjets for cuts
       sort ( subjets.begin(), subjets.end(), GreaterByPtCandPtr() );
       
       // Now look at the subjets that were formed
       for ( int isub = 0; isub < 2; ++isub ) {

	 // Get this subjet
	 Jet::Constituent icandJet = subjets[isub];

	 // Now look at the "other" subjets than this one, form the minimum invariant mass
	 // pairing, as well as the "closest" combination to the W mass
	 for ( int jsub = isub + 1; jsub < 3; ++jsub ) {

	   // Get the second subjet
	   Jet::Constituent jcandJet = subjets[jsub];
	 
	   // Form a W candidate out of the two jets
	   CompositeCandidate wCand("wCand");
	   wCand.addDaughter( *icandJet, "jet1" );
	   wCand.addDaughter( *jcandJet, "jet2" );

	   // Add the four momenta
	   addFourMomenta.set(wCand);

	   // Get the candidate mass
	   double imw = wCand.mass();

	   // Find the combination closest to the W mass
	   if ( fabs( imw - WMass_ ) < properties.wMass ) {
	     properties.wMass = imw;
	   }
	   // Find the minimum mass pairing. 
	   if ( fabs( imw ) < properties.minMass ) {
	     properties.minMass = imw;
	   }
	   // Print out some useful information
	   if ( verbose_ ) {
	     cout << "Creating W candidates, examining: " << endl;
	     cout << "icand = " << isub << ", pt = " << icandJet->pt() << ", eta = " << icandJet->eta() << ", phi = " << icandJet->phi() << endl;
	     cout << "jcand = " << jsub << ", pt = " << jcandJet->pt() << ", eta = " << jcandJet->eta() << ", phi = " << jcandJet->phi() << endl;
	   }
	   
	 }// end second loop over subjets
       }// end first loop over subjets
     }// endif 3 subjets
     
     CATopJetTagInfo tagInfo;
     tagInfo.insert( ref, properties );
     tagInfos->push_back( tagInfo );
   }// end loop over hard jets
  
   iEvent.put( tagInfos );
 
   return;   
}


// ------------ method called once each job just before starting event loop  ------------
void 
CATopJetTagger::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CATopJetTagger::endJob() {

}

//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetTagger);
