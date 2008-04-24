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
// $Id: V0Fitter.cc,v 1.22 2008/04/24 17:59:09 drell Exp $
//
//

#include "RecoVertex/V0Producer/interface/V0Fitter.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include <typeinfo>

// Constants

const double piMass = 0.13957018;
const double piMassSquared = piMass*piMass;

// Constructor and (empty) destructor
V0Fitter::V0Fitter(const edm::ParameterSet& theParameters,
		   const edm::Event& iEvent, const edm::EventSetup& iSetup) :
  recoAlg(theParameters.getUntrackedParameter("trackRecoAlgorithm",
				   std::string("ctfWithMaterialTracks"))) {
  using std::string;

  // ------> Initialize parameters from PSet. ALL TRACKED, so no defaults.
  // First set bits to do various things:
  //  -decide whether to use the KVF track smoother, and whether to store those
  //     tracks in the reco::Vertex
  useRefTrax = theParameters.getParameter<bool>(string("useSmoothing"));
  storeRefTrax = theParameters.getParameter<bool>(
				string("storeSmoothedTracksInRecoVertex"));

  //  -whether to reconstruct K0s
  doKshorts = theParameters.getParameter<bool>(string("selectKshorts"));
  //  -whether to reconstruct Lambdas
  doLambdas = theParameters.getParameter<bool>(string("selectLambdas"));

  //  -whether to do cuts or store all found V0 
  doPostFitCuts = theParameters.getParameter<bool>(string("doPostFitCuts"));
  doTkQualCuts = 
    theParameters.getParameter<bool>(string("doTrackQualityCuts"));

  // Second, initialize post-fit cuts
  chi2Cut = theParameters.getParameter<double>(string("vtxChi2Cut"));
  tkChi2Cut = theParameters.getParameter<double>(string("tkChi2Cut"));
  tkNhitsCut = theParameters.getParameter<int>(string("tkNhitsCut"));
  rVtxCut = theParameters.getParameter<double>(string("rVtxCut"));
  vtxSigCut = theParameters.getParameter<double>(string("vtxSignificanceCut"));
  collinCut = theParameters.getParameter<double>(string("collinearityCut"));
  kShortMassCut = theParameters.getParameter<double>(string("kShortMassCut"));
  lambdaMassCut = theParameters.getParameter<double>(string("lambdaMassCut"));

  // FOR DEBUG:
  //initFileOutput();
  //--------------------

  //std::cout << "Entering V0Producer" << std::endl;

  fitAll(iEvent, iSetup);

  // FOR DEBUG:
  //cleanupFileOutput();
  //--------------------

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
  vector<TrackRef> theTrackRefs_;//temporary, for pre-cut TrackRefs
  vector<Track> theTracks;
  vector<TrackRef> theTrackRefs;
  vector<TransientTrack> theTransTracks;

  // Handles for tracks, B-field, and tracker geometry
  Handle<reco::TrackCollection> theTrackHandle;
  ESHandle<MagneticField> bFieldHandle;
  ESHandle<TrackerGeometry> trackerGeomHandle;
  ESHandle<GlobalTrackingGeometry> globTkGeomHandle;


  // Get the tracks from the event, and get the B-field record
  //  from the EventSetup
  iEvent.getByLabel(recoAlg, theTrackHandle);
  if( !theTrackHandle->size() ) return;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeomHandle);
  iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

  //edm::ESHandle<TransientTrackBuilder> transTkBuilder;
  //iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", transTkBuilder);

  trackerGeom = trackerGeomHandle.product();
  magField = bFieldHandle.product();

  // Create a vector of TrackRef objects to store in reco::Vertex later
  for(unsigned int indx = 0; indx < theTrackHandle->size(); indx++) {
    TrackRef tmpRef( theTrackHandle, indx );
    theTrackRefs_.push_back( tmpRef );
  }

  // Beam spot.  I'm hardcoding to (0,0,0) right now, but will use 
  //  reco::BeamSpot later.
  reco::TrackBase::Point beamSpot(0,0,0);

  //
  //----->> Preselection cuts on tracks.
  //

  // Check track refs, look at closest approach point to beam spot,
  //  and fill track vector with ones that pass the cut (> 1 cm).


  // REMOVE THIS CUT, AND MAKE A SCATTER PLOT OF MASS VS. LARGEST IMPACT
  //  PARAMETER OF CHARGED DAUGHTER TRACK.

  for( unsigned int indx2 = 0; indx2 < theTrackRefs_.size(); indx2++ ) {
    TransientTrack tmpTk( *(theTrackRefs_[indx2]), &(*bFieldHandle), globTkGeomHandle );
    //TransientTrack tmpTk( transTkBuilder->build( theTrackRefs_[indx2] ) );
    /*TrajectoryStateClosestToBeamLine
      tscb( tmpTk.stateAtBeamLine() );
      std::cout << "Didn't fail on tscb creation." << std::endl;*/
    //if( tscb.transverseImpactParameter().value() > 0.1 ) {
  //    if(tscb.transverseImpactParameter().value() > 0.) {//This removes the cut.
      theTrackRefs.push_back( theTrackRefs_[indx2] );
      theTracks.push_back( *(theTrackRefs_[indx2]) );
      theTransTracks.push_back( tmpTk );
      //    }
  }

  //----->> Initial cuts finished.

  // Loop over tracks and vertex good charged track pairs
  for(unsigned int trdx1 = 0; trdx1 < theTracks.size(); trdx1++) {

    if( doTkQualCuts 
	&& theTrackRefs[trdx1]->normalizedChi2() > tkChi2Cut )
      continue;

    if( doTkQualCuts && theTrackRefs[trdx1]->recHitsSize() ) {
      trackingRecHit_iterator tk1HitIt = theTrackRefs[trdx1]->recHitsBegin();
      int nHits1 = 0;
      for( ; tk1HitIt < theTrackRefs[trdx1]->recHitsEnd(); tk1HitIt++) {
	if( (*tk1HitIt)->isValid() ) nHits1++;
      }
      if( nHits1 < tkNhitsCut ) continue;
    }
    
    for(unsigned int trdx2 = trdx1 + 1; trdx2 < theTracks.size(); trdx2++) {

      if( doTkQualCuts 
	  && theTrackRefs[trdx2]->normalizedChi2() > tkChi2Cut ) continue;

      if( doTkQualCuts && theTrackRefs[trdx2]->recHitsSize() ) {
	trackingRecHit_iterator tk2HitIt = theTrackRefs[trdx2]->recHitsBegin();
	int nHits2 = 0;
	for( ; tk2HitIt < theTrackRefs[trdx2]->recHitsEnd(); tk2HitIt++) {
	  if( (*tk2HitIt)->isValid() ) nHits2++;
	}
	if( nHits2 < tkNhitsCut ) continue;
      }

      vector<TransientTrack> transTracks;

      TrackRef positiveTrackRef;
      TrackRef negativeTrackRef;
      Track* positiveIter = 0;
      Track* negativeIter = 0;
      TransientTrack* posTransTkPtr = 0;
      TransientTrack* negTransTkPtr = 0;

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
      /*      double posESq = positiveIter->momentum().Mag2() + piMassSquared;
      double negESq = negativeIter->momentum().Mag2() + piMassSquared;
      double posE = sqrt(posESq);
      double negE = sqrt(negESq);
      double totalE = posE + negE;
      double totalESq = totalE*totalE;
      double totalPSq = 
	(positiveIter->momentum() + negativeIter->momentum()).Mag2();
      double mass = sqrt( totalESq - totalPSq);

      TrajectoryStateClosestToBeamLine
	tscbPos( posTransTkPtr->stateAtBeamLine() );
      double d0_pos = tscbPos.transverseImpactParameter().value();
      TrajectoryStateClosestToBeamLine
	tscbNeg( negTransTkPtr->stateAtBeamLine() );
      double d0_neg = tscbNeg.transverseImpactParameter().value();
      */

      //std::cout << "Calculated m-pi-pi: " << mass << std::endl;
      /*mPiPiMassOut << mass << " "
	<< (d0_neg < d0_pos? d0_pos : d0_neg) << std::endl;*/
      //if( mass > 0.7 ) continue;

      //----->> Finished making cuts.

      // Create TransientTrack objects.  They're needed for the KVF.

      // Fill the vector of TransientTracks to send to KVF
      transTracks.push_back(*posTransTkPtr);
      transTracks.push_back(*negTransTkPtr);

      //posTransTkPtr = negTransTkPtr = 0;
      positiveIter = negativeIter = 0;

      // Create the vertex fitter object
      KalmanVertexFitter theFitter(useRefTrax == 0 ? false : true);

      // Vertex the tracks
      //CachingVertex theRecoVertex;
      TransientVertex theRecoVertex;
      theRecoVertex = theFitter.vertex(transTracks);

      bool continue_ = true;
    
      // If the vertex is valid, make a VertexCompositeCandidate with it
      //  to be stored in the Event if the vertex Chi2 < 20

      if( !theRecoVertex.isValid() ) {
	continue_ = false;
      }

      if( continue_ ) {
	if(theRecoVertex.totalChiSquared() > 20. 
	   || theRecoVertex.totalChiSquared() < 0.) {
	  continue_ = false;
	}
      }


      if( continue_ ) {
	// Create reco::Vertex object to be put into the Event
	reco::Vertex theVtx = theRecoVertex;
	/*
	std::cout << "bef: reco::Vertex: " << theVtx.tracksSize() 
		  << " " << theVtx.hasRefittedTracks() << std::endl;
	std::cout << "bef: reco::TransientVertex: " 
		  << theRecoVertex.originalTracks().size() 
		  << " " << theRecoVertex.hasRefittedTracks() << std::endl;
	*/
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
	//if(storeRefTrax) {
	for( ; traxIter != traxEnd; traxIter++) {
	  if( traxIter->track().charge() > 0. ) {
	    thePositiveRefTrack = new TransientTrack(*traxIter);
	  }
	  else if (traxIter->track().charge() < 0.) {
	    theNegativeRefTrack = new TransientTrack(*traxIter);
	  }
	}
	//}
	/*else {
	  theVtx.add( positiveTrackRef );
	  theVtx.add( negativeTrackRef );
	  }*/
	/*
	std::cout << "aft: reco::Vertex: " << theVtx.tracksSize() 
		  << " " << theVtx.hasRefittedTracks() << std::endl;
	std::cout << "aft: reco::TransientVertex: " 
		  << theRecoVertex.originalTracks().size() 
		  << " " << theRecoVertex.hasRefittedTracks() << std::endl
		  << std::endl;
	*/
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
			 posTransTkPtr->trajectoryStateClosestToPoint(vtxPos));
	  trajMins = new TrajectoryStateClosestToPoint(
			 negTransTkPtr->trajectoryStateClosestToPoint(vtxPos));

	}

	posTransTkPtr = negTransTkPtr = 0;

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
	const Particle::LorentzVector kShortP4(totalP.x(), 
					 totalP.y(), totalP.z(), 
					 kShortETot);
	const Particle::LorentzVector lambdaP4(totalP.x(), 
					 totalP.y(), totalP.z(), 
					 lambdaEtot);
	const Particle::LorentzVector lambdaBarP4(totalP.x(), 
					 totalP.y(), totalP.z(), 
					 lambdaBarEtot);
	Particle::Point vtx(theVtx.x(), theVtx.y(), theVtx.z());
	const Vertex::CovarianceMatrix vtxCov(theVtx.covariance());
	double vtxChi2(theVtx.chi2());
	double vtxNdof(theVtx.ndof());

	// Create the VertexCompositeCandidate object that will be stored in the Event
	VertexCompositeCandidate 
	  theKshort(0, kShortP4, vtx, vtxCov, vtxChi2, vtxNdof);
	VertexCompositeCandidate 
	  theLambda(0, lambdaP4, vtx, vtxCov, vtxChi2, vtxNdof);
	VertexCompositeCandidate 
	  theLambdaBar(0, lambdaBarP4, vtx, vtxCov, vtxChi2, vtxNdof);

	// Create daughter candidates for the VertexCompositeCandidates
	RecoChargedCandidate 
	  thePiPlusCand(1, Particle::LorentzVector(positiveP.x(), 
						positiveP.y(), positiveP.z(),
						piPlusE), vtx);
	thePiPlusCand.setTrack(positiveTrackRef);

	RecoChargedCandidate
	  thePiMinusCand(-1, Particle::LorentzVector(negativeP.x(), 
						 negativeP.y(), negativeP.z(),
						 piMinusE), vtx);
	thePiMinusCand.setTrack(negativeTrackRef);

 
	RecoChargedCandidate
	  theProtonCand(1, Particle::LorentzVector(positiveP.x(),
						 positiveP.y(), positiveP.z(),
						 protonE), vtx);
	theProtonCand.setTrack(positiveTrackRef);

	RecoChargedCandidate
	  theAntiProtonCand(-1, Particle::LorentzVector(negativeP.x(),
						 negativeP.y(), negativeP.z(),
						 antiProtonE), vtx);
	theAntiProtonCand.setTrack(negativeTrackRef);


	// Store the daughter Candidates in the VertexCompositeCandidates
	theKshort.addDaughter(thePiPlusCand);
	theKshort.addDaughter(thePiMinusCand);

	theLambda.addDaughter(theProtonCand);
	theLambda.addDaughter(thePiMinusCand);

	theLambdaBar.addDaughter(theAntiProtonCand);
	theLambdaBar.addDaughter(thePiPlusCand);

	theKshort.setPdgId(310);
	theLambda.setPdgId(3122);
	theLambdaBar.setPdgId(-3122);

	AddFourMomenta addp4;
	addp4.set( theKshort );
	addp4.set( theLambda );
	addp4.set( theLambdaBar );

	// Store the candidates in a temporary STL vector
	if(doKshorts)
	  preCutCands.push_back(theKshort);
	if(doLambdas) {
	  preCutCands.push_back(theLambda);
	  preCutCands.push_back(theLambdaBar);
	}


      }
    }
  }

  // If we have any candidates, clean and sort them into proper collections
  //  using the cuts specified in the EventSetup.
  if( preCutCands.size() )
    applyPostFitCuts();

}


// Get methods
reco::VertexCompositeCandidateCollection V0Fitter::getKshorts() const {
  return theKshorts;
}

reco::VertexCompositeCandidateCollection V0Fitter::getLambdas() const {
  return theLambdas;
}

reco::VertexCompositeCandidateCollection V0Fitter::getLambdaBars() const {
  return theLambdaBars;
}


// Method that applies cuts to the vector of pre-cut candidates.
void V0Fitter::applyPostFitCuts() {
  //static int eventCounter = 1;
  //std::cout << "Doing applyPostFitCuts()" << std::endl;
  /*std::cout << "Starting post fit cuts with " << preCutCands.size()
    << " preCutCands" << std::endl;*/
  //std::cout << "!1" << std::endl;
  for(reco::VertexCompositeCandidateCollection::iterator theIt = preCutCands.begin();
      theIt != preCutCands.end(); theIt++) {
    bool writeVee = false;
    double rVtxMag = sqrt( theIt->vertex().x()*theIt->vertex().x() +
			   theIt->vertex().y()*theIt->vertex().y() );
			   //theIt->vertex().z()*theIt->vertex().z() );

    double x_ = theIt->vertex().x();
    double y_ = theIt->vertex().y();
    //double z_ = theIt->vertex().z();
    double sig00 = theIt->vertexCovariance(0,0);
    double sig11 = theIt->vertexCovariance(1,1);
    //double sig22 = theIt->vertexCovariance(2,2);
    double sig01 = theIt->vertexCovariance(0,1);
    //double sig02 = theIt->vertexCovariance(0,2);
    //double sig12 = theIt->vertexCovariance(1,2);

    /*double sigmaRvtxMag =
      sqrt( sig00*(x_*x_) + sig11*(y_*y_) + sig22*(z_*z_)
	    + 2*(sig01*(x_*y_) + sig02*(x_*z_) + sig12*(y_*z_)) ) 
	    / rVtxMag;*/
    double sigmaRvtxMag =
      sqrt( sig00*(x_*x_) + sig11*(y_*y_) + 2*sig01*(x_*y_) ) / rVtxMag;

    //std::cout << "!2" << std::endl;
    // Get the tracks from the candidates.
    std::vector<reco::RecoChargedCandidate> v0daughters;
    std::vector<reco::TrackRef> theVtxTrax;
    for(unsigned int ii = 0; ii < theIt->numberOfDaughters(); ii++) {
      v0daughters.push_back( *(dynamic_cast<reco::RecoChargedCandidate *>
			       (theIt->daughter(ii))) );
    }
    for(unsigned int jj = 0; jj < v0daughters.size(); jj++) {
      theVtxTrax.push_back(v0daughters[jj].track());
    }
    /*    if(theIt->vertex().hasRefittedTracks()) {
      theVtxTrax = theIt->vertex().refittedTracks();
    }
    else {
      reco::Vertex::trackRef_iterator theTkIt = theIt->vertex().tracks_begin();
      for( ; theTkIt < theIt->vertex().tracks_end(); theTkIt++) {
	reco::TrackRef theRef1 = theTkIt->castTo<reco::TrackRef>();
	theVtxTrax.push_back(*theRef1);
      }
      }*/

    //std::cout << "!3" << std::endl;
    using namespace reco;

    // If the position of the innermost hit on either of the daughter
    //   tracks is less than the radial vertex position (minus 4 sigmaRvtx)
    //   then don't keep the vee.
    bool hitsOkay = true;
    //std::cout << "theVtxTrax.size = " << theVtxTrax.size() << std::endl;
    if( theVtxTrax.size() == 2 && doPostFitCuts) {
      if( theVtxTrax[0]->innerOk() ) {
	reco::Vertex::Point tk1HitPosition = theVtxTrax[0]->innerPosition();
	if( sqrt(tk1HitPosition.Perp2()) < (rVtxMag - sigmaRvtxMag*4.0) ) {
	  hitsOkay = false;
	}
      }
      if( theVtxTrax[1]->innerOk() && hitsOkay) {
	reco::Vertex::Point tk2HitPosition = theVtxTrax[1]->innerPosition();
	if( sqrt(tk2HitPosition.Perp2()) < (rVtxMag - sigmaRvtxMag*4.0) ) {
	  hitsOkay = false;
	}
      }
      /*if( theVtxTrax[0]->recHitsSize() && theVtxTrax[1]->recHitsSize() ) {

	trackingRecHit_iterator tk1HitIt = theVtxTrax[0]->recHitsBegin();
	trackingRecHit_iterator tk2HitIt = theVtxTrax[1]->recHitsBegin();

	for( ; tk1HitIt < theVtxTrax[0]->recHitsEnd(); tk1HitIt++) {
	  if( (*tk1HitIt)->isValid() && hitsOkay) {
	    const TrackingRecHit* tk1HitPtr = (*tk1HitIt).get();
	    GlobalPoint tk1HitPosition
	      = trackerGeom->idToDet(tk1HitPtr->
				     geographicalId())->
	      surface().toGlobal(tk1HitPtr->localPosition());
	    //std::cout << typeid(*tk1HitPtr).name();

	    if( tk1HitPosition.perp() < (rVtxMag - 4.*sigmaRvtxMag) ) {
	      hitsOkay = false;
	    }
	  }
	}

	for( ; tk2HitIt < theVtxTrax[1]->recHitsEnd(); tk2HitIt++) {
	  if( (*tk2HitIt)->isValid() && hitsOkay) {
	    const TrackingRecHit* tk2HitPtr = (*tk2HitIt).get();
	    GlobalPoint tk2HitPosition
	      = trackerGeom->idToDet(tk2HitPtr->
				     geographicalId())->
	      surface().toGlobal(tk2HitPtr->localPosition());

	    if( tk2HitPosition.perp() < (rVtxMag - 4.*sigmaRvtxMag) ) {
	      hitsOkay = false;
	    }
	  }
	}
	}*/
    }


    if( theIt->vertexChi2() < chi2Cut &&
	rVtxMag > rVtxCut &&
	rVtxMag/sigmaRvtxMag > vtxSigCut &&
	hitsOkay) {
      writeVee = true;
    }
    const double kShortMass = 0.49767;
    const double lambdaMass = 1.1156;

    if( theIt->mass() < kShortMass + kShortMassCut && 
	theIt->mass() > kShortMass - kShortMassCut && writeVee &&
	doKshorts && doPostFitCuts) {
      if(theIt->pdgId() == 310) {
	theKshorts.push_back( *theIt );
      }
    }
    else if( !doPostFitCuts && theIt->pdgId() == 310 && doKshorts ) {
      theKshorts.push_back( *theIt );
    }

    //3122
    else if( theIt->mass() < lambdaMass + lambdaMassCut &&
	theIt->mass() > lambdaMass - lambdaMassCut && writeVee &&
	     doLambdas && doPostFitCuts ) {
      if(theIt->pdgId() == 3122) {
	theLambdas.push_back( *theIt );
      }
      else if(theIt->pdgId() == -3122) {
	theLambdaBars.push_back( *theIt );
      }
    }
    else if ( !doPostFitCuts && theIt->pdgId() == 3122 && doLambdas ) {
      theLambdas.push_back( *theIt );
    }
    else if ( !doPostFitCuts && theIt->pdgId() == -3122 && doLambdas ) {
      theLambdaBars.push_back( *theIt );
    }
  }

  /*static int numkshorts = 0;
  numkshorts += theKshorts.size();
  std::cout << "Ending cuts with " << theKshorts.size() << " K0s, "
  << numkshorts << " total." << std::endl;*/
  //std::cout << "Finished applyPostFitCuts() for event "
  //    << eventCounter << std::endl;
  //eventCounter++;


}

