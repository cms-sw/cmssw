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
// $Id: V0Fitter.cc,v 1.39 2009/06/09 23:28:49 drell Exp $
//
//

#include "RecoVertex/V0Producer/interface/V0Fitter.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <typeinfo>

// Constants

const double piMass = 0.13957018;
const double piMassSquared = piMass*piMass;
const double protonMassSquared = 0.880354402;
const double kShortMass = 0.49767;
const double lambdaMass = 1.1156;

// Constructor and (empty) destructor
V0Fitter::V0Fitter(const edm::ParameterSet& theParameters,
		   const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using std::string;

  // Get the track reco algorithm from the ParameterSet
  recoAlg = theParameters.getParameter<edm::InputTag>("trackRecoAlgorithm");

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
  impactParameterSigCut = theParameters.getParameter<double>(string("impactParameterSigCut"));
  mPiPiCut = theParameters.getParameter<double>(string("mPiPiCut"));
  tkDCACut = theParameters.getParameter<double>(string("tkDCACut"));
  vtxFitter = theParameters.getParameter<edm::InputTag>("vertexFitter");

  //edm::LogInfo("V0Producer") << "Using " << vtxFitter << " to fit V0 vertices.\n";
  //std::cout << "Using " << vtxFitter << " to fit V0 vertices." << std::endl;
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
  using std::cout;
  using std::endl;
  using reco::Track;
  using reco::TransientTrack;
  using reco::TrackRef;
  using namespace edm;

  //cout << "In fitAll()" << endl;

  // Create std::vectors for Tracks and TrackRefs (required for
  //  passing to the KalmanVertexFitter)
  vector<TrackRef> theTrackRefs;
  vector<TransientTrack> theTransTracks;

  // Handles for tracks, B-field, and tracker geometry
  Handle<reco::TrackCollection> theTrackHandle;
  Handle<reco::BeamSpot> theBeamSpotHandle;
  ESHandle<MagneticField> bFieldHandle;
  ESHandle<TrackerGeometry> trackerGeomHandle;
  ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
  //cout << "Check 0" << endl;

  // Get the tracks from the event, and get the B-field record
  //  from the EventSetup
  iEvent.getByLabel(recoAlg, theTrackHandle);
  iEvent.getByLabel(std::string("offlineBeamSpot"), theBeamSpotHandle);
  if( !theTrackHandle->size() ) return;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeomHandle);
  iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

  //edm::ESHandle<TransientTrackBuilder> transTkBuilder;
  //iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", transTkBuilder);

  trackerGeom = trackerGeomHandle.product();
  magField = bFieldHandle.product();

  // Get the beam spot position, width, and error on the width
  /*reco::BeamSpot thePrimary;
  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByType(beamSpotHandle);
  thePrimary = *beamSpotHandle;
  reco::BeamSpot::Point primaryPos( thePrimary.position() );
  double beamWidth = thePrimary.BeamWidth();
  double beamWidthError = thePrimary.BeamWidthError();*/

  // Fill vectors of TransientTracks and TrackRefs after applying preselection cuts.
  for(unsigned int indx = 0; indx < theTrackHandle->size(); indx++) {
    TrackRef tmpRef( theTrackHandle, indx );

    if( doTkQualCuts && 
        tmpRef->normalizedChi2() < tkChi2Cut &&
        tmpRef->numberOfValidHits() >= tkNhitsCut ) {
      TransientTrack tmpTk( *tmpRef, &(*bFieldHandle), globTkGeomHandle );
      TrajectoryStateClosestToBeamLine tscb( tmpTk.stateAtBeamLine() );
      if( tscb.isValid() ) {
	if( tscb.transverseImpactParameter().significance() > impactParameterSigCut ) {
	  theTrackRefs.push_back( tmpRef );
	  theTransTracks.push_back( tmpTk );
	}
      }
    }
    else if ( !doTkQualCuts ) {
      TransientTrack tmpTk2( *tmpRef, &(*bFieldHandle), globTkGeomHandle );
      theTrackRefs.push_back( tmpRef );
      theTransTracks.push_back( tmpTk2 );
    }

  }

  // Good tracks have now been selected for vertexing.  Move on to vertex fitting.


  // Loop over tracks and vertex good charged track pairs
  for(unsigned int trdx1 = 0; trdx1 < theTrackRefs.size(); trdx1++) {

    for(unsigned int trdx2 = trdx1 + 1; trdx2 < theTrackRefs.size(); trdx2++) {

      //This vector holds the pair of oppositely-charged tracks to be vertexed
      vector<TransientTrack> transTracks;

      TrackRef positiveTrackRef;
      TrackRef negativeTrackRef;
      TransientTrack* posTransTkPtr = 0;
      TransientTrack* negTransTkPtr = 0;

      // Look at the two tracks we're looping over.  If they're oppositely
      //  charged, load them into the hypothesized positive and negative tracks
      //  and references to be sent to the KalmanVertexFitter
      if(theTrackRefs[trdx1]->charge() < 0. && 
	 theTrackRefs[trdx2]->charge() > 0.) {
	negativeTrackRef = theTrackRefs[trdx1];
	positiveTrackRef = theTrackRefs[trdx2];
	negTransTkPtr = &theTransTracks[trdx1];
	posTransTkPtr = &theTransTracks[trdx2];
      }
      else if(theTrackRefs[trdx1]->charge() > 0. &&
	      theTrackRefs[trdx2]->charge() < 0.) {
	negativeTrackRef = theTrackRefs[trdx2];
	positiveTrackRef = theTrackRefs[trdx1];
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
            
      /*double posESq = positiveTrackRef->momentum().Mag2() + piMassSquared;
      double negESq = negativeTrackRef->momentum().Mag2() + piMassSquared;
      double posE = sqrt(posESq);
      double negE = sqrt(negESq);
      double totalE = posE + negE;
      double totalESq = totalE*totalE;
      double totalPSq = 
	(positiveTrackRef->momentum() + negativeTrackRef->momentum()).Mag2();
      double mass = sqrt( totalESq - totalPSq);
      if( mass > mPiPiCut ) continue;*/
      
      //^^^Next, need to make sure the above works with signal/background studies

      //----->> Finished making cuts.

      // Fill the vector of TransientTracks to send to KVF
      transTracks.push_back(*posTransTkPtr);
      transTracks.push_back(*negTransTkPtr);

      // Trajectory states to calculate DCA for the 2 tracks
      //cout << "Check 1" << endl;
      FreeTrajectoryState posState = posTransTkPtr->impactPointTSCP().theState();
      FreeTrajectoryState negState = negTransTkPtr->impactPointTSCP().theState();

      if( !posTransTkPtr->impactPointTSCP().isValid() || !negTransTkPtr->impactPointTSCP().isValid() ) continue;
      //cout << "Check 2" << endl;

      // Measure distance between tracks at their closest approach
      ClosestApproachInRPhi cApp;
      cApp.calculate(posState, negState);
      //cout << cApp.status() << endl;
      float dca = fabs( cApp.distance() );
      GlobalPoint cxPt = cApp.crossingPoint();

      if (sqrt( cxPt.x()*cxPt.x() + cxPt.y()*cxPt.y() ) > 120. 
          || cxPt.z() > 300.) continue;
      if (dca < 0. || dca > tkDCACut) continue;

      // Get trajectory states for the tracks at POCA for later cuts
      //cout << "crossingPoint: " << cxPt.x() << ", " << cxPt.y() << ", " << cxPt.z() << endl;
      //cout << "Check 3" << endl;
      //TrajectoryStateClosestToPoint posTSCP;
      //TrajectoryStateClosestToPoint negTSCP;
      //try {
      TrajectoryStateClosestToPoint posTSCP = 
	posTransTkPtr->trajectoryStateClosestToPoint( cxPt );
      TrajectoryStateClosestToPoint negTSCP =
	negTransTkPtr->trajectoryStateClosestToPoint( cxPt );

      if( !posTSCP.isValid() || !negTSCP.isValid() ) continue;
      //}
      //catch(...) {
      //cout << "Caught one." << endl;
      //continue;
      //}
      //cout << "Check 4" << endl;

 

      //double posESq = positiveTrackRef->momentum().Mag2() + piMassSquared;
      //double negESq = negativeTrackRef->momentum().Mag2() + piMassSquared;
      double posESq = posTSCP.momentum().mag2() + piMassSquared;
      double negESq = negTSCP.momentum().mag2() + piMassSquared;
      double posE = sqrt(posESq);
      double negE = sqrt(negESq);
      double totalE = posE + negE;
      double totalESq = totalE*totalE;
      //double totalPSq = 
      //(positiveTrackRef->momentum() + negativeTrackRef->momentum()).Mag2();
      double totalPSq =
	( posTSCP.momentum() + negTSCP.momentum() ).mag2();
      double mass = sqrt( totalESq - totalPSq);

      //mPiPiMassOut << mass << std::endl;

      if( mass > mPiPiCut ) continue;

      // Create the vertex fitter object and vertex the tracks
      TransientVertex theRecoVertex;
      if(vtxFitter == string("KalmanVertexFitter")) {
	KalmanVertexFitter theKalmanFitter(useRefTrax == 0 ? false : true);
	theRecoVertex = theKalmanFitter.vertex(transTracks);
      }
      else if (vtxFitter == string("AdaptiveVertexFitter")) {
	useRefTrax = storeRefTrax = false;
	AdaptiveVertexFitter theAdaptiveFitter;
	theRecoVertex = theAdaptiveFitter.vertex(transTracks);
      }

      // Vertex the tracks
      //TransientVertex theRecoVertex;
      //theRecoVertex = theFitter.vertex(transTracks);
    
      // If the vertex is valid, make a VertexCompositeCandidate with it

      if( !theRecoVertex.isValid() || theRecoVertex.totalChiSquared() < 0. ) {
	continue;
      }

      // Create reco::Vertex object for use in creating the Candidate
      reco::Vertex theVtx = theRecoVertex;
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
        
      // 
      for( ; traxIter != traxEnd; ++traxIter) {
	if( traxIter->track().charge() > 0. ) {
	  thePositiveRefTrack = new TransientTrack(*traxIter);
	}
	else if (traxIter->track().charge() < 0.) {
	  theNegativeRefTrack = new TransientTrack(*traxIter);
	}
      }

      // Do post-fit cuts if specified in config file.

      // Find the vertex d0 and its error

      typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3> > SMatrixSym3D;
      typedef ROOT::Math::SVector<double, 3> SVector3;

      GlobalPoint vtxPos(theVtx.x(), theVtx.y(), theVtx.z());

      //      double dummy = findV0MassError(vtxPos, transTracks);
      //double x_ = vtxPos.x();
      //double y_ = vtxPos.y();
      //double rVtxMag = sqrt( x_*x_ + y_*y_);
      //double sig00 = theVtx.covariance(0,0);
      //double sig11 = theVtx.covariance(1,1);
      //double sig01 = theVtx.covariance(0,1);
      //double sigmaRvtxMag =
      //sqrt( sig00*(x_*x_) + sig11*(y_*y_) + 2*sig01*(x_*y_) ) / rVtxMag;

      GlobalPoint beamSpotPos(theBeamSpotHandle->position().x(),
			      theBeamSpotHandle->position().y(),
			      theBeamSpotHandle->position().z());
      //double beam_x = beamSpotPos.x();
      //double beam_y = beamSpotPos.y();
      //reco::BeamSpot::Covariance3DMatrix spotCov = theBeamSpotHandle->covariance3D();
      //reco::Vertex::CovarianceMatrix vtxCov = theVtx.covariance();

      SMatrixSym3D totalCov = theBeamSpotHandle->covariance3D() + theVtx.covariance();
      SVector3 distanceVector(vtxPos.x() - beamSpotPos.x(),
			      vtxPos.y() - beamSpotPos.y(),
			      0.);//so that we get radial values only, 
                                  //since z beamSpot uncertainty is huge

      double rVtxMag = ROOT::Math::Mag(distanceVector);
      double sigmaRvtxMag = sqrt(ROOT::Math::Similarity(totalCov, distanceVector)) / rVtxMag;

      // The methods innerOk() and innerPosition() require TrackExtra, which
      // is only available in the RECO data tier, not AOD. This may be a problem.
      if( positiveTrackRef->innerOk() ) {
	reco::Vertex::Point posTkHitPos = positiveTrackRef->innerPosition();
	if( sqrt( posTkHitPos.Perp2() ) < ( rVtxMag - sigmaRvtxMag*4. ) 
	    && doPostFitCuts ) {
	  if(thePositiveRefTrack) delete thePositiveRefTrack;
	  if(theNegativeRefTrack) delete theNegativeRefTrack;
	  thePositiveRefTrack = theNegativeRefTrack = 0;
	  continue;
	}
      }
      if( negativeTrackRef->innerOk() ) {
	reco::Vertex::Point negTkHitPos = negativeTrackRef->innerPosition();
	if( sqrt( negTkHitPos.Perp2() ) < ( rVtxMag - sigmaRvtxMag*4. ) 
	    && doPostFitCuts ) {
	  if(thePositiveRefTrack) delete thePositiveRefTrack;
	  if(theNegativeRefTrack) delete theNegativeRefTrack;
	  thePositiveRefTrack = theNegativeRefTrack = 0;
	  continue;
	}
      }

      if( doPostFitCuts ) {
	if( theVtx.normalizedChi2() > chi2Cut ||
	    rVtxMag < rVtxCut ||
	    rVtxMag / sigmaRvtxMag < vtxSigCut ) {
	  if(thePositiveRefTrack) delete thePositiveRefTrack;
	  if(theNegativeRefTrack) delete theNegativeRefTrack;
	  thePositiveRefTrack = theNegativeRefTrack = 0;
	  continue;
	}
      }

      // Cuts finished, now we create the candidates and push them back into the collections.
      
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

      if( !trajPlus->isValid() || !trajMins->isValid() ) continue;

      posTransTkPtr = negTransTkPtr = 0;

      GlobalVector positiveP(trajPlus->momentum());
      GlobalVector negativeP(trajMins->momentum());
      GlobalVector totalP(positiveP + negativeP);

      //double posP = positiveP.mag();
      //double negP = negativeP.mag();

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
			    + piMassSquared);
      double piMinusE = sqrt(negativeP.x()*negativeP.x()
			     + negativeP.y()*negativeP.y()
			     + negativeP.z()*negativeP.z()
			     + piMassSquared);
      double protonE = sqrt(positiveP.x()*positiveP.x()
			    + positiveP.y()*positiveP.y()
			    + positiveP.z()*positiveP.z()
			    + protonMassSquared);
      double antiProtonE = sqrt(negativeP.x()*negativeP.x()
				+ negativeP.y()*negativeP.y()
				+ negativeP.z()*negativeP.z()
				+ protonMassSquared);
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
      VertexCompositeCandidate* theKshort = 0;
      VertexCompositeCandidate* theLambda = 0;
      VertexCompositeCandidate* theLambdaBar = 0;

      if( doKshorts ) {
	theKshort = new VertexCompositeCandidate(0, kShortP4, vtx, vtxCov, vtxChi2, vtxNdof);
      }
      if( doLambdas ) {
	if( positiveP.mag() > negativeP.mag() ) {
	  theLambda = 
	    new VertexCompositeCandidate(0, lambdaP4, vtx, vtxCov, vtxChi2, vtxNdof);
	}
	else {
	  theLambdaBar = 
	    new VertexCompositeCandidate(0, lambdaBarP4, vtx, vtxCov, vtxChi2, vtxNdof);
	}
      }

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


      AddFourMomenta addp4;
      // Store the daughter Candidates in the VertexCompositeCandidates 
      //    if they pass mass cuts
      if( doKshorts ) {
	theKshort->addDaughter(thePiPlusCand);
	theKshort->addDaughter(thePiMinusCand);
	theKshort->setPdgId(310);
	addp4.set( *theKshort );
	if( doPostFitCuts &&
	    theKshort->mass() < kShortMass + kShortMassCut &&
	    theKshort->mass() > kShortMass - kShortMassCut ) {
	  theKshorts.push_back( *theKshort );
	}
	else if (!doPostFitCuts) {
	  theKshorts.push_back( *theKshort );
	}
      }
      
      if( doLambdas && theLambda ) {
	theLambda->addDaughter(theProtonCand);
	theLambda->addDaughter(thePiMinusCand);
	theLambda->setPdgId(3122);
	addp4.set( *theLambda );
	if( doPostFitCuts &&
	    theLambda->mass() < lambdaMass + lambdaMassCut &&
	    theLambda->mass() > lambdaMass - lambdaMassCut ) {
	  theLambdas.push_back( *theLambda );
	}
	else if (!doPostFitCuts) {
	  theLambdas.push_back( *theLambda );
	}
      }
      else if ( doLambdas && theLambdaBar ) {
	theLambdaBar->addDaughter(theAntiProtonCand);
	theLambdaBar->addDaughter(thePiPlusCand);
	theLambdaBar->setPdgId(-3122);
	addp4.set( *theLambdaBar );
	if( doPostFitCuts &&
	    theLambdaBar->mass() < lambdaMass + lambdaMassCut &&
	    theLambdaBar->mass() > lambdaMass - lambdaMassCut ) {
	  //	  theLambdaBars.push_back( *theLambdaBar );
	  theLambdas.push_back( *theLambdaBar );
	}
	else if (!doPostFitCuts) {
	  //	  theLambdaBars.push_back( *theLambdaBar );
	  theLambdas.push_back( *theLambdaBar );
	}
      }

      if(theKshort) delete theKshort;
      if(theLambda) delete theLambda;
      if(theLambdaBar) delete theLambdaBar;
      theKshort = theLambda = theLambdaBar = 0;

    }
  }
}

double V0Fitter::findV0MassError(const GlobalPoint &vtxPos, std::vector<reco::TransientTrack> dauTracks) { 
  return -1.;
}

/*
double V0Fitter::findV0MassError(const GlobalPoint &vtxPos, std::vector<reco::TransientTrack> dauTracks) {
  // Returns -99999. if trajectory states fail at vertex position

  // Load positive track trajectory at vertex into vector, then negative track
  std::vector<TrajectoryStateClosestToPoint> sortedTrajStatesAtVtx;
  for( unsigned int ndx = 0; ndx < dauTracks.size(); ndx++ ) {
    if( dauTracks[ndx].trajectoryStateClosestToPoint(vtxPos).isValid() ) {
      std::cout << "From TSCP: " 
		<< dauTracks[ndx].trajectoryStateClosestToPoint(vtxPos).perigeeParameters().transverseCurvature()
		<< "; From Track: " << dauTracks[ndx].track().qoverp() << std::endl;
    }
    if( sortedTrajStatesAtVtx.size() == 0 ) {
      if( dauTracks[ndx].charge() > 0 ) {
	sortedTrajStatesAtVtx.push_back( dauTracks[ndx].trajectoryStateClosestToPoint(vtxPos) );
      }
      else {
	sortedTrajStatesAtVtx.push_back( dauTracks[ndx].trajectoryStateClosestToPoint(vtxPos) );
      }
    }
  }
  std::vector<PerigeeTrajectoryParameters> param;
  std::vector<PerigeeTrajectoryError> paramError;
  std::vector<GlobalVector> momenta;

  for( unsigned int ndx2 = 0; ndx2 < sortedTrajStatesAtVtx.size(); ndx2++ ) {
    if( sortedTrajStatesAtVtx[ndx2].isValid() ) {
      param.push_back( sortedTrajStatesAtVtx[ndx2].perigeeParameters() );
      paramError.push_back( sortedTrajStatesAtVtx[ndx2].perigeeError() );
      momenta.push_back( sortedTrajStatesAtVtx[ndx2].momentum() );
    }
    else return -99999.;
  }
  return 0;
}
*/


// Get methods
const reco::VertexCompositeCandidateCollection& V0Fitter::getKshorts() const {
  return theKshorts;
}

const reco::VertexCompositeCandidateCollection& V0Fitter::getLambdas() const {
  return theLambdas;
}

//const reco::VertexCompositeCandidateCollection& V0Fitter::getLambdaBars() const {
//  return theLambdaBars;
//}


// Method that applies cuts to the vector of pre-cut candidates.
/*
void V0Fitter::applyPostFitCuts() {

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
}

*/
