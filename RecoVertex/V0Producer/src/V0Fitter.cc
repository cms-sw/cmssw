// -*- C++ -*-
//
// Package:    V0Producer
// Class:      V0Fitter
// 
/**\class V0Fitter V0Fitter.cc RecoVertex/V0Producer/src/V0Fitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Fri May 18 22:57:40 CEST 2007
// $Id: V0Fitter.cc,v 1.1 2007/07/05 12:25:41 drell Exp $
//
//

#include "RecoVertex/V0Producer/interface/V0Fitter.h"

// Constructor and (empty) destructor
V0Fitter::V0Fitter(const edm::Event& iEvent, const edm::EventSetup& iSetup,
		   std::string trackRecoAlgo, const int useRefittedTrax,
		   const int storeRefittedTrax, const double chi2Cut_,
		   const double rVtxCut_, const double vtxSigCut_,
		   const double collinCut_, const double kShortMassCut_,
		   const double lambdaMassCut_,
		   const int doK0s, const int doLam) :
  recoAlg(trackRecoAlgo) {
  useRefTrax = useRefittedTrax;
  storeRefTrax = storeRefittedTrax;
  doKshorts = doK0s;
  doLambdas = doLam;

  // Initialize cut values
  chi2Cut = chi2Cut_;
  rVtxCut = rVtxCut_;
  vtxSigCut = vtxSigCut_;
  collinCut = collinCut_;
  kShortMassCut = kShortMassCut_;
  lambdaMassCut = lambdaMassCut_;

  fitAll(iEvent, iSetup);
}

V0Fitter::~V0Fitter() {
}

// Method containing the algorithm for vertex reconstruction
void V0Fitter::fitAll(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using std::vector;
  using reco::Track;
  using reco::TransientTrack;
  using reco::TrackRef;
  using namespace edm;

  // Create std::vectors for Tracks and TrackRefs (required for
  //  passing to the KalmanVertexFitter)
  vector<Track> theTracks;
  vector<TrackRef> theTrackRefs;

  // Handles for tracks and B-field
  Handle<reco::TrackCollection> theTrackHandle;
  ESHandle<MagneticField> bFieldHandle;


  // Get the tracks from the event, and get the B-field record
  //  from the EventSetup
  iEvent.getByLabel(recoAlg, theTrackHandle);
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);


  // Fill our std::vector<Track> with the reconstructed tracks from
  //  the handle
  theTracks.insert( theTracks.end(), theTrackHandle->begin(),
		    theTrackHandle->end() );

  // Create a vector of TrackRef objects to store in reco::Vertex later
  for(unsigned int indx = 0; indx < theTrackHandle->size(); indx++) {
    theTrackRefs.push_back( TrackRef(theTrackHandle, indx) );
  }

  // Call private method that does initial cutting of the reconstructed
  //  tracks.  Passes theTracks by reference.
  applyPreFitCuts(theTracks);

  // Loop over tracks and vertex good charged track pairs
  for(unsigned int trdx1 = 0; trdx1 < theTracks.size(); trdx1++) {
    for(unsigned int trdx2 = trdx1 + 1; trdx2 < theTracks.size(); trdx2++) {
      
      vector<TransientTrack> transTracks;

      TrackRef positiveTrackRef;
      TrackRef negativeTrackRef;
      Track* positiveIter = 0;
      Track* negativeIter = 0;

      // Look at the two tracks we're looping over.  If they're oppositely
      //  charged, load them into the hypothesized positive and negative tracks
      //  and references to be sent to the KalmanVertexFitter
      if(theTrackRefs[trdx1]->charge() < 0. && 
	 theTrackRefs[trdx2]->charge() > 0.) {
	negativeTrackRef = theTrackRefs[trdx1];
	positiveTrackRef = theTrackRefs[trdx2];
	negativeIter = &theTracks[trdx1];
	positiveIter = &theTracks[trdx2];
      }
      else if(theTrackRefs[trdx1]->charge() > 0. &&
	      theTrackRefs[trdx2]->charge() < 0.) {
	negativeTrackRef = theTrackRefs[trdx2];
	positiveTrackRef = theTrackRefs[trdx1];
	negativeIter = &theTracks[trdx2];
	positiveIter = &theTracks[trdx1];
      }
      // If they're not 2 oppositely charged tracks, loop back to the
      //  beginning and try the next pair.
      else continue;

      // Create TransientTrack objects.  They're needed for the KVF.
      TransientTrack thePositiveTransTrack( *positiveIter, &(*bFieldHandle) );
      TransientTrack theNegativeTransTrack( *negativeIter, &(*bFieldHandle) );

      // Fill the vector of TransientTracks to send to KVF
      transTracks.push_back(thePositiveTransTrack);
      transTracks.push_back(theNegativeTransTrack);

      // Create the vertex fitter object
      KalmanVertexFitter theFitter(useRefTrax == 0 ? false : true);

      // Vertex the tracks
      //CachingVertex theRecoVertex;
      TransientVertex theRecoVertex;
      theRecoVertex = theFitter.vertex(transTracks);


      // If the vertex is valid, make a V0Candidate with it
      //  to be stored in the Event
      //  Also implementing a chi2 cut of 20.
      bool continue_ = true;
      if( !theRecoVertex.isValid() ) {
	continue_ = false;
      }

      if( continue_ ) {
	if(theRecoVertex.totalChiSquared() > 20. ) {
	  continue_ = false;
	}
      }

      if( continue_ ) {
	// Create reco::Vertex object to be put into the Event
	TransientVertex tempVtx = theRecoVertex;
	reco::Vertex theVtx = tempVtx;

	// Create and fill vector of refitted TransientTracks
	//  (iff they've been created by the KVF)
	vector<TransientTrack> refittedTrax;
	if( tempVtx.hasRefittedTracks() ) {
	  refittedTrax = tempVtx.refittedTracks();
	}
	// Need an iterator over the refitted tracks for below
	vector<TransientTrack>::iterator traxIter = refittedTrax.begin(),
	  traxEnd = refittedTrax.end();

	// TransientTrack objects to hold the positive and negative
	//  refitted tracks
	TransientTrack* thePositiveRefTrack = 0;
	TransientTrack* theNegativeRefTrack = 0;
	
	// Store track info in reco::Vertex object.  If the option is set,
	//  store the refitted ones as well.
	if(storeRefTrax) {
	  for( ; traxIter != traxEnd; traxIter++) {
	    if( traxIter->track().charge() > 0. ) {
	      theVtx.add( positiveTrackRef, traxIter->track(), 1.);
	      thePositiveRefTrack = new TransientTrack(*traxIter);
	    }
	    else if (traxIter->track().charge() < 0.) {
	      theVtx.add( negativeTrackRef, traxIter->track(), 1.);
	      theNegativeRefTrack = new TransientTrack(*traxIter);
	    }
	  }
	}
	else {
	  theVtx.add( positiveTrackRef );
	  theVtx.add( negativeTrackRef );
	}

	// Calculate momentum vectors for both tracks at the vertex
	GlobalPoint vtxPos(theVtx.x(), theVtx.y(), theVtx.z());

	TrajectoryStateClosestToPoint* trajPlus;
	TrajectoryStateClosestToPoint* trajMins;

	if(useRefTrax && refittedTrax.size() > 1) {
	  trajPlus = new TrajectoryStateClosestToPoint(
		  thePositiveRefTrack->trajectoryStateClosestToPoint(vtxPos));
	  trajMins = new TrajectoryStateClosestToPoint(
		  theNegativeRefTrack->trajectoryStateClosestToPoint(vtxPos));
	}
	else {
	  trajPlus = new TrajectoryStateClosestToPoint(
		  thePositiveTransTrack.trajectoryStateClosestToPoint(vtxPos));
	  trajMins = new TrajectoryStateClosestToPoint(
		  theNegativeTransTrack.trajectoryStateClosestToPoint(vtxPos));
	}

	//  Just fixed this to make candidates for all 3 V0 particle types.
	//  NOT TESTED AT ALL.
	//   We'll also need to make sure we're not writing candidates
	//    for all 3 types for a single event.  This could be problematic..
	GlobalVector positiveP(trajPlus->momentum());
	GlobalVector negativeP(trajMins->momentum());
	GlobalVector totalP(positiveP + negativeP);
	double piMassSq = 0.019479101;
	double protonMassSq = 0.880262585374;

	//cleanup stuff we don't need anymore
	delete trajPlus;
	delete trajMins;
	trajPlus = trajMins = 0;
	delete thePositiveRefTrack;
	delete theNegativeRefTrack;
	thePositiveRefTrack = theNegativeRefTrack = 0;

	// calculate total energy of V0 3 ways:
	//  Assume it's a kShort, a Lambda, or a LambdaBar.
	double piPlusE = sqrt(positiveP.x()*positiveP.x()
			      + positiveP.y()*positiveP.y()
			      + positiveP.z()*positiveP.z()
			      + piMassSq);
	double piMinusE = sqrt(negativeP.x()*negativeP.x()
			       + negativeP.y()*negativeP.y()
			       + negativeP.z()*negativeP.z()
			       + piMassSq);
	double protonE = sqrt(positiveP.x()*positiveP.x()
			      + positiveP.y()*positiveP.y()
			      + positiveP.z()*positiveP.z()
			      + protonMassSq);
	double antiProtonE = sqrt(negativeP.x()*negativeP.x()
				  + negativeP.y()*negativeP.y()
				  + negativeP.z()*negativeP.z()
				  + protonMassSq);
	double kShortETot = piPlusE + piMinusE;
	double lambdaEtot = protonE + piMinusE;
	double lambdaBarEtot = antiProtonE + piPlusE;

	using namespace reco;

	// Create momentum 4-vectors for the 3 candidate types
	Particle::LorentzVector kShortP4(totalP.x(), 
					 totalP.y(), totalP.z(), 
					 kShortETot);
	Particle::LorentzVector lambdaP4(totalP.x(), 
					 totalP.y(), totalP.z(), 
					 lambdaEtot);
	Particle::LorentzVector lambdaBarP4(totalP.x(), 
					 totalP.y(), totalP.z(), 
					 lambdaBarEtot);
	Particle::Point vtx(theVtx.x(), theVtx.y(), theVtx.z());

	// Create the V0Candidate object that will be stored in the Event
	V0Candidate theKshort(0, kShortP4, Particle::Point(0,0,0));
	V0Candidate theLambda(0, lambdaP4, Particle::Point(0,0,0));
	V0Candidate theLambdaBar(0, lambdaBarP4, Particle::Point(0,0,0));
	// The above lines are hardcoded for the origin.  Need to fix.

	// Set the V0Candidates' vertex to the one we found above
	//  (and loaded with track info)
	theKshort.setVertex(theVtx);
	theLambda.setVertex(theVtx);
	theLambdaBar.setVertex(theVtx);


	// Create daughter candidates for the V0Candidates
	RecoChargedCandidate 
	  thePiPlusCand(1, Particle::LorentzVector(positiveP.x(), 
						positiveP.y(), positiveP.z(),
						piPlusE), vtx);
	thePiPlusCand.setTrack(positiveTrackRef);

	RecoChargedCandidate
	  theProtonCand(1, Particle::LorentzVector(positiveP.x(),
						  positiveP.y(), positiveP.z(),
						  protonE), vtx);
	theProtonCand.setTrack(positiveTrackRef);

	RecoChargedCandidate
	  thePiMinusCand(-1, Particle::LorentzVector(negativeP.x(), 
						 negativeP.y(), negativeP.z(),
						 piMinusE), vtx);
	thePiMinusCand.setTrack(negativeTrackRef);

	RecoChargedCandidate
	  theAntiProtonCand(-1, Particle::LorentzVector(negativeP.x(),
						 negativeP.y(), negativeP.z(),
						 antiProtonE), vtx);
	theAntiProtonCand.setTrack(negativeTrackRef);

	// Store the daughter Candidates in the V0Candidates
	theKshort.addDaughter(thePiPlusCand);
	theKshort.addDaughter(thePiMinusCand);

	theLambda.addDaughter(theProtonCand);
	theLambda.addDaughter(thePiMinusCand);

	theLambdaBar.addDaughter(theAntiProtonCand);
	theLambdaBar.addDaughter(thePiPlusCand);

	theKshort.setPdgId(310);
	theLambda.setPdgId(3122);
	theLambdaBar.setPdgId(-3122);

	// Store the candidates in a temporary STL vector
	preCutCands.push_back(theKshort);
	preCutCands.push_back(theLambda);
	preCutCands.push_back(theLambdaBar);
      }
    }
  }

  applyPostFitCuts();

}


// Get methods
std::vector<reco::V0Candidate> V0Fitter::getKshorts() const {
  return theKshorts;
}

std::vector<reco::V0Candidate> V0Fitter::getLambdas() const {
  return theLambdas;
}

std::vector<reco::V0Candidate> V0Fitter::getLambdaBars() const {
  return theLambdaBars;
}

void V0Fitter::applyPreFitCuts(std::vector<reco::Track> &tracks) {

}

void V0Fitter::applyPostFitCuts() {
  for(std::vector<reco::V0Candidate>::iterator theIt = preCutCands.begin();
      theIt != preCutCands.end(); theIt++) {
    bool writeVee = false;
    double rVtxMag = sqrt( theIt->vertex().x()*theIt->vertex().x() +
			   theIt->vertex().y()*theIt->vertex().y() +
			   theIt->vertex().z()*theIt->vertex().z() );
    double sigmaRvtxMag = 
      sqrt( theIt->vertex().xError()*theIt->vertex().xError() *
	    theIt->vertex().yError()*theIt->vertex().yError() *
	    theIt->vertex().zError()*theIt->vertex().zError() );
    if( theIt->vertex().chi2() < chi2Cut &&
	rVtxMag > rVtxCut &&
	rVtxMag/sigmaRvtxMag > vtxSigCut ) {
      writeVee = true;
    }
    const double kShortMass = 0.49767;
    const double lambdaMass = 1.1156;
    if( theIt->mass() < kShortMass + kShortMassCut && 
	theIt->mass() > kShortMass - kShortMassCut && writeVee &&
	doKshorts) {
      //theIt->setPdgId(310);
      theKshorts.push_back( *theIt );
    }
    //3122
    else if( theIt->mass() < lambdaMass + lambdaMassCut &&
	theIt->mass() > lambdaMass - lambdaMassCut && writeVee &&
	     doLambdas) {
      //theIt->setPdgId(3122);
      if(theIt->pdgId() > 0) {
	theLambdas.push_back( *theIt );
      }
      else {
	theLambdaBars.push_back( *theIt );
      }
    }
  }

}


/*
reco::VertexCollection V0Fitter::getKshortCollection() const {
  return K0s;
}

reco::VertexCollection V0Fitter::getLambdaCollection() const {
  return Lam0;
}

reco::VertexCollection V0Fitter::getLambdaBarCollection() const {
  return Lam0Bar;
}
*/
