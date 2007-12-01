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
// $Id: V0Fitter.cc,v 1.8 2007/09/19 05:34:54 drell Exp $
//
//

#include "RecoVertex/V0Producer/interface/V0Fitter.h"

#include <typeinfo>

// Constants

const double piMass = 0.13957018;
const double piMassSquared = piMass*piMass;

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
  //vector<Track> theTracks_;
  vector<TrackRef> theTrackRefs_;
  vector<TrackRef> theTrackRefs;
  vector<TransientTrack> theTransTracks;

  // Handles for tracks, B-field, and tracker geometry
  Handle<reco::TrackCollection> theTrackHandle;
  ESHandle<MagneticField> bFieldHandle;
  ESHandle<TrackerGeometry> trackerGeomHandle;


  // Get the tracks from the event, and get the B-field record
  //  from the EventSetup
  iEvent.getByLabel(recoAlg, theTrackHandle);
  if( !theTrackHandle->size() ) return;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeomHandle);

  trackerGeom = trackerGeomHandle.product();

  // Create a vector of TrackRef objects to store in reco::Vertex later
  for(unsigned int indx = 0; indx < theTrackHandle->size(); indx++) {
    TrackRef tmpRef( theTrackHandle, indx );
    theTrackRefs_.push_back( tmpRef );
    //theTracks_.push_back( *tmpRef );
  }

  //std::cout << "@@@@ " << theTrackHandle->size() << std::endl;

  // Fill our std::vector<Track> with the reconstructed tracks from
  //  the handle
  /*  theTracks.insert( theTracks.end(), theTrackHandle->begin(),
      theTrackHandle->end() );*/

  // Beam spot.  I'm hardcoding to (0,0,0) right now, but will use 
  //  reco::BeamSpot later.
  reco::TrackBase::Point beamSpot(0,0,0);

  //
  //----->> Preselection cuts on tracks.
  //

  // Check track refs, look at closest approach point to beam spot,
  //  and fill track vector with ones that pass the cut (> 1 cm).

  for( unsigned int indx2 = 0; indx2 < theTrackRefs_.size(); indx2++ ) {
    TransientTrack tmpTk( *(theTrackRefs_[indx2]), &(*bFieldHandle) );
    TrajectoryStateClosestToBeamLine
      tscb( tmpTk.stateAtBeamLine() );
    if( tscb.transverseImpactParameter().value() > 0.1 ) {
    /*if( sqrt(  theTrackRefs_[indx2]->dxy( beamSpot )
	     * theTrackRefs_[indx2]->dxy( beamSpot )
	     + theTrackRefs_[indx2]->dsz( beamSpot )
	     * theTrackRefs_[indx2]->dsz( beamSpot ) ) > 0.5 ) {*/
      theTrackRefs.push_back( theTrackRefs_[indx2] );
      theTracks.push_back( *(theTrackRefs_[indx2]) );
      theTransTracks.push_back( tmpTk );
    }
  }

  // UNUSED:
  // Call private method that does initial cutting of the reconstructed
  //  tracks.  Passes theTracks by reference.
  //applyPreFitCuts(theTracks);

  //----->> Initial cuts finished.


  // Loop over tracks and vertex good charged track pairs
  for(unsigned int trdx1 = 0; trdx1 < theTracks.size(); trdx1++) {
    for(unsigned int trdx2 = trdx1 + 1; trdx2 < theTracks.size(); 
	trdx2++) {
      //vector<Track> theTracks;  
      vector<TransientTrack> transTracks;

      //theTracks.push_back( theTransTracks[trdx1].track() );
      //theTracks.push_back( theTransTracks[trdx2].track() );

      TrackRef positiveTrackRef;
      TrackRef negativeTrackRef;
      Track* positiveIter = 0;
      Track* negativeIter = 0;
      TransientTrack* posTransTkPtr;
      TransientTrack* negTransTkPtr;

      // Look at the two tracks we're looping over.  If they're oppositely
      //  charged, load them into the hypothesized positive and negative tracks
      //  and references to be sent to the KalmanVertexFitter
      if(theTrackRefs[trdx1]->charge() < 0. && 
	 theTrackRefs[trdx2]->charge() > 0.) {
	negativeTrackRef = theTrackRefs[trdx1];
	positiveTrackRef = theTrackRefs[trdx2];
	negativeIter = &theTracks[trdx1];
	positiveIter = &theTracks[trdx2];
	negTransTkPtr = &theTransTracks[trdx1];
	posTransTkPtr = &theTransTracks[trdx2];
      }
      else if(theTrackRefs[trdx1]->charge() > 0. &&
	      theTrackRefs[trdx2]->charge() < 0.) {
	negativeTrackRef = theTrackRefs[trdx2];
	positiveTrackRef = theTrackRefs[trdx1];
	negativeIter = &theTracks[trdx2];
	positiveIter = &theTracks[trdx1];
	negTransTkPtr = &theTransTracks[trdx2];
	posTransTkPtr = &theTransTracks[trdx1];
      }
      // If they're not 2 oppositely charged tracks, loop back to the
      //  beginning and try the next pair.
      else continue;

      //
      //----->> Carry out in-loop preselection cuts on tracks.
      //

      // Assume pion masses and do a wide mass cut.  First, we need the
      //  track momenta.
      double posESq = positiveIter->momentum().Mag2() + piMassSquared;
      double negESq = negativeIter->momentum().Mag2() + piMassSquared;
      double posE = sqrt(posESq);
      double negE = sqrt(negESq);
      double totalE = posE + negE;
      double totalESq = totalE*totalE;
      double totalPSq = 
	(positiveIter->momentum() + negativeIter->momentum()).Mag2();
      double mass = sqrt( totalESq - totalPSq);

      //std::cout << "Calculated m-pi-pi: " << mass << std::endl;

      //----->> Finished making cuts.

      // Create TransientTrack objects.  They're needed for the KVF.
      //TransientTrack thePositiveTransTrack( *positiveIter, &(*bFieldHandle) );
      //TransientTrack theNegativeTransTrack( *negativeIter, &(*bFieldHandle) );

      // Fill the vector of TransientTracks to send to KVF
      //transTracks.push_back(thePositiveTransTrack);
      //transTracks.push_back(theNegativeTransTrack);
      transTracks.push_back(*posTransTkPtr);
      transTracks.push_back(*negTransTkPtr);

      // Create the vertex fitter object
      KalmanVertexFitter theFitter(useRefTrax == 0 ? false : true);

      // Vertex the tracks
      //CachingVertex theRecoVertex;
      TransientVertex theRecoVertex;
      theRecoVertex = theFitter.vertex(transTracks);

      bool continue_ = true;
      
      // ADDED 8-27, loop over RecHits on both tracks to see if the position
      //  is inside that of the reconstructed vertex.  If it is, chuck the Vee.

      bool yesorno = false;

      /*if( thePositiveTransTrack.recHitsSize() 
	 && theNegativeTransTrack.recHitsSize() ) {
	trackingRecHit_iterator posTrackHitIt 
	  = thePositiveTransTrack.recHitsBegin();
	trackingRecHit_iterator negTrackHitIt
	= theNegativeTransTrack.recHitsBegin();*/
      if( posTransTkPtr->recHitsSize()
	  && negTransTkPtr->recHitsSize() ) {
	trackingRecHit_iterator posTrackHitIt
	  = posTransTkPtr->recHitsBegin();
	trackingRecHit_iterator negTrackHitIt
	  = negTransTkPtr->recHitsBegin();
	
	//std::cout << "Doing the recHit thing." << std::endl;
	
	//for( ; posTrackHitIt < thePositiveTransTrack.recHitsEnd();
	for( ; posTrackHitIt < posTransTkPtr->recHitsEnd();
	     posTrackHitIt++) {
	  const TrackingRecHit* posHitPtr = (*posTrackHitIt).get();
	  if( (*posTrackHitIt)->isValid() && theRecoVertex.isValid() ) {
	    GlobalPoint posHitPosition 
	      = trackerGeom->idToDet((*posTrackHitIt)->
				     geographicalId())->
	      surface().toGlobal((*posTrackHitIt)->localPosition());

	    //std::cout << "@@POS: " << posHitPosition.perp() << std::endl;
	  
	    if( posHitPosition.perp() < theRecoVertex.position().perp() ) {
	      /*std::cout << typeid(*posHitPtr).name()
		<< "+" << theRecoVertex.position().perp() -
		posHitPosition.perp() 
		<< std::endl;*/
	      yesorno = true;
	    }
	  }
	}
	
	//for( ; negTrackHitIt < theNegativeTransTrack.recHitsEnd();
	for( ; negTrackHitIt < negTransTkPtr->recHitsEnd();
	     negTrackHitIt++) {
	  const TrackingRecHit* negHitPtr = (*negTrackHitIt).get();
	  if( (*negTrackHitIt)->isValid() && theRecoVertex.isValid() ) {
	    GlobalPoint negHitPosition 
	      = trackerGeom->idToDet((*negTrackHitIt)->
				     geographicalId())->
	      surface().toGlobal((*negTrackHitIt)->localPosition());
	    
	    //std::cout << "@@NEG: " << negHitPosition.perp() << std::endl;
	    
	    if( negHitPosition.perp() < theRecoVertex.position().perp() ) {
	      /*std::cout << typeid(*negHitPtr).name()
		<< "-" << theRecoVertex.position().perp() -
		negHitPosition.perp()
		<< std::endl;*/
	      yesorno = true;
	    }
	  }
	}
      }
      if(yesorno) {
	//std::cout << "End of track pair hits." << std::endl;
      }
      
      // If the vertex is valid, make a V0Candidate with it
      //  to be stored in the Event
      //  Also implementing a chi2 cut of 20.

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
	//TransientVertex tempVtx = theRecoVertex;
	reco::Vertex theVtx = theRecoVertex;

	std::cout << "bef: reco::Vertex: " << theVtx.tracksSize() 
		  << " " << theVtx.hasRefittedTracks() << std::endl;
	std::cout << "bef: reco::TransientVertex: " 
		  << theRecoVertex.originalTracks().size() 
		  << " " << theRecoVertex.hasRefittedTracks() << std::endl;

	// Create and fill vector of refitted TransientTracks
	//  (iff they've been created by the KVF)
	vector<TransientTrack> refittedTrax;
	if( theRecoVertex.hasRefittedTracks() ) {
	  refittedTrax = theRecoVertex.refittedTracks();
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
	/*float one_ = 1.;
	if(storeRefTrax) {
	  for( ; traxIter != traxEnd; traxIter++) {
	    if( traxIter->track().charge() > 0. ) {
	      theVtx.add( positiveTrackRef, traxIter->track(), one_ ); 
	      thePositiveRefTrack = new TransientTrack(*traxIter);
	    }
	    else if (traxIter->track().charge() < 0.) {
	      theVtx.add( negativeTrackRef, traxIter->track(), one_);
	      theNegativeRefTrack = new TransientTrack(*traxIter);
	    }
	  }
	}
	else {
	  theVtx.add( positiveTrackRef );
	  theVtx.add( negativeTrackRef );
	  }*/

	std::cout << "aft: reco::Vertex: " << theVtx.tracksSize() 
		  << " " << theVtx.hasRefittedTracks() << std::endl;
	std::cout << "aft: reco::TransientVertex: " 
		  << theRecoVertex.originalTracks().size() 
		  << " " << theRecoVertex.hasRefittedTracks() << std::endl
		  << std::endl;

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
		//thePositiveTransTrack.trajectoryStateClosestToPoint(vtxPos));
			 posTransTkPtr->trajectoryStateClosestToPoint(vtxPos));
	  trajMins = new TrajectoryStateClosestToPoint(
		//theNegativeTransTrack.trajectoryStateClosestToPoint(vtxPos));
			 negTransTkPtr->trajectoryStateClosestToPoint(vtxPos));

	}

	//  Just fixed this to make candidates for all 3 V0 particle types.
	// 
	//   We'll also need to make sure we're not writing candidates
	//    for all 3 types for a single event.  This could be problematic..
	GlobalVector positiveP(trajPlus->momentum());
	GlobalVector negativeP(trajMins->momentum());
	GlobalVector totalP(positiveP + negativeP);
	double piMassSq = 0.019479835;
	double protonMassSq = 0.880354402;

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

  if( preCutCands.size() )
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
    // WRONG!

    /*    double sigmaRvtxMag = 
      sqrt( theIt->vertex().xError()*theIt->vertex().xError() *
	    theIt->vertex().yError()*theIt->vertex().yError() *
	    theIt->vertex().zError()*theIt->vertex().zError() );*/

    double x_ = theIt->vertex().x();
    double y_ = theIt->vertex().y();
    double z_ = theIt->vertex().z();
    double sig00 = theIt->vertex().covariance(0,0);
    double sig11 = theIt->vertex().covariance(1,1);
    double sig22 = theIt->vertex().covariance(2,2);
    double sig01 = theIt->vertex().covariance(0,1);
    double sig02 = theIt->vertex().covariance(0,2);
    double sig12 = theIt->vertex().covariance(1,2);

    double sigmaRvtxMag =
      sqrt( sig00*(x_*x_) + sig11*(y_*y_) + sig22*(z_*z_)
	    + 2*(sig01*(x_*y_) + sig02*(x_*z_) + sig12*(y_*z_)) ) 
      / rVtxMag;

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
      if(theIt->pdgId() == 310) {
	theKshorts.push_back( *theIt );
      }
    }
    //3122
    else if( theIt->mass() < lambdaMass + lambdaMassCut &&
	theIt->mass() > lambdaMass - lambdaMassCut && writeVee &&
	     doLambdas) {
      //theIt->setPdgId(3122);
      if(theIt->pdgId() == 3122) {
	theLambdas.push_back( *theIt );
      }
      else if(theIt->pdgId() == -3122) {
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
