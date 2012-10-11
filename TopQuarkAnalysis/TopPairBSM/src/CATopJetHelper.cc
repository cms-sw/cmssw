#include "CATopJetHelper.h"


struct GreaterByPtCandPtr {
  bool operator()( const edm::Ptr<reco::Candidate> & t1, const edm::Ptr<reco::Candidate> & t2 ) const {
    return t1->pt() > t2->pt();
  }
};


reco::CATopJetProperties CATopJetHelper::operator()( reco::Jet const & ihardJet ) const {
  reco::CATopJetProperties properties;
  // Get subjets
  reco::Jet::Constituents subjets = ihardJet.getJetConstituents();
  properties.nSubJets = subjets.size();  // number of subjets
  properties.topMass = ihardJet.mass();      // jet mass
  properties.wMass = 99999.;                  // best W mass
  properties.minMass = 999999.;            // minimum mass pairing

  // Require at least three subjets in all cases, if not, untagged
  if ( properties.nSubJets >= 3 ) {

    // Take the highest 3 pt subjets for cuts
    sort ( subjets.begin(), subjets.end(), GreaterByPtCandPtr() );
       
    // Now look at the subjets that were formed
    for ( int isub = 0; isub < 2; ++isub ) {

      // Get this subjet
      reco::Jet::Constituent icandJet = subjets[isub];

      // Now look at the "other" subjets than this one, form the minimum invariant mass
      // pairing, as well as the "closest" combination to the W mass
      for ( int jsub = isub + 1; jsub < 3; ++jsub ) {

	// Get the second subjet
	reco::Jet::Constituent jcandJet = subjets[jsub];

	reco::Candidate::LorentzVector wCand = icandJet->p4() + jcandJet->p4();

	// Get the candidate mass
	double imw = wCand.mass();

	// Find the combination closest to the W mass
	if ( fabs( imw - WMass_ ) < fabs(properties.wMass - WMass_) ) {
	  properties.wMass = imw;
	}
	// Find the minimum mass pairing. 
	if ( fabs( imw ) < properties.minMass ) {
	  properties.minMass = imw;
	}  
      }// end second loop over subjets
    }// end first loop over subjets
  }// endif 3 subjets
 
  if (properties.minMass == 999999){properties.minMass=-1;}

  return properties;
}
