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
// $Id: V0Fitter.cc,v 1.57 2013/04/11 23:17:44 wmtan Exp $
//
//

#include "RecoVertex/V0Producer/interface/V0Fitter.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <typeinfo>
#include <memory>

// Constants

const double piMass = 0.13957018;
const double piMassSquared = piMass*piMass;
const double protonMass = 0.938272013;
const double protonMassSquared = protonMass*protonMass;
const double kShortMass = 0.497614;
const double lambdaMass = 1.115683;

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

  //  -whether to reconstruct K0s
  doKshorts = theParameters.getParameter<bool>(string("selectKshorts"));
  //  -whether to reconstruct Lambdas
  doLambdas = theParameters.getParameter<bool>(string("selectLambdas"));

  // Second, initialize post-fit cuts
  chi2Cut = theParameters.getParameter<double>(string("vtxChi2Cut"));
  tkChi2Cut = theParameters.getParameter<double>(string("tkChi2Cut"));
  tkNhitsCut = theParameters.getParameter<int>(string("tkNhitsCut"));
  rVtxCut = theParameters.getParameter<double>(string("rVtxCut"));
  vtxSigCut = theParameters.getParameter<double>(string("vtxSignificance2DCut"));
  collinCut = theParameters.getParameter<double>(string("collinearityCut"));
  kShortMassCut = theParameters.getParameter<double>(string("kShortMassCut"));
  lambdaMassCut = theParameters.getParameter<double>(string("lambdaMassCut"));
  impactParameterSigCut = theParameters.getParameter<double>(string("impactParameterSigCut"));
  mPiPiCut = theParameters.getParameter<double>(string("mPiPiCut"));
  tkDCACut = theParameters.getParameter<double>(string("tkDCACut"));
  vtxFitter = theParameters.getParameter<edm::InputTag>("vertexFitter");
  innerHitPosCut = theParameters.getParameter<double>(string("innerHitPosCut"));
  std::vector<std::string> qual = theParameters.getParameter<std::vector<std::string> >("trackQualities");
  for (unsigned int ndx = 0; ndx < qual.size(); ndx++) {
    qualities.push_back(reco::TrackBase::qualityByName(qual[ndx]));
  }

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
  using namespace reco;
  using namespace edm;

  // Create std::vectors for Tracks and TrackRefs (required for
  //  passing to the KalmanVertexFitter)
  std::vector<TrackRef> theTrackRefs;
  std::vector<TransientTrack> theTransTracks;

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

  trackerGeom = trackerGeomHandle.product();
  magField = bFieldHandle.product();

  // Fill vectors of TransientTracks and TrackRefs after applying preselection cuts.
  for(unsigned int indx = 0; indx < theTrackHandle->size(); indx++) {
    TrackRef tmpRef( theTrackHandle, indx );
    bool quality_ok = true;
    if (qualities.size()!=0) {
      quality_ok = false;
      for (unsigned int ndx_ = 0; ndx_ < qualities.size(); ndx_++) {
	if (tmpRef->quality(qualities[ndx_])){
	  quality_ok = true;
	  break;          
	}
      }
    }
    if( !quality_ok ) continue;


    if( tmpRef->normalizedChi2() < tkChi2Cut &&
        tmpRef->numberOfValidHits() >= tkNhitsCut ) {
      TransientTrack tmpTk( *tmpRef, &(*bFieldHandle), globTkGeomHandle );
      
      FreeTrajectoryState initialFTS = trajectoryStateTransform::initialFreeState(*tmpRef, magField);
      TSCBLBuilderNoMaterial blsBuilder;
      TrajectoryStateClosestToBeamLine tscb( blsBuilder(initialFTS, *theBeamSpotHandle) );
      
      if( tscb.isValid() ) {
	if( tscb.transverseImpactParameter().significance() > impactParameterSigCut ) {
	  theTrackRefs.push_back( tmpRef );
	  theTransTracks.push_back( tmpTk );
	}
      }
    }
  }

  // Good tracks have now been selected for vertexing.  Move on to vertex fitting.


  // Loop over tracks and vertex good charged track pairs
  for(unsigned int trdx1 = 0; trdx1 < theTrackRefs.size(); trdx1++) {

    for(unsigned int trdx2 = trdx1 + 1; trdx2 < theTrackRefs.size(); trdx2++) {

      //This vector holds the pair of oppositely-charged tracks to be vertexed
      std::vector<TransientTrack> transTracks;

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

      // Fill the vector of TransientTracks to send to KVF
      transTracks.push_back(*posTransTkPtr);
      transTracks.push_back(*negTransTkPtr);

      // Trajectory states to calculate DCA for the 2 tracks
      FreeTrajectoryState posState = posTransTkPtr->impactPointTSCP().theState();
      FreeTrajectoryState negState = negTransTkPtr->impactPointTSCP().theState();

      if( !posTransTkPtr->impactPointTSCP().isValid() || !negTransTkPtr->impactPointTSCP().isValid() ) continue;

      // Measure distance between tracks at their closest approach
      ClosestApproachInRPhi cApp;
      cApp.calculate(posState, negState);
      if( !cApp.status() ) continue;
      float dca = fabs( cApp.distance() );
      GlobalPoint cxPt = cApp.crossingPoint();

      if (dca < 0. || dca > tkDCACut) continue;
      if (sqrt( cxPt.x()*cxPt.x() + cxPt.y()*cxPt.y() ) > 120. 
          || std::abs(cxPt.z()) > 300.) continue;

      // Get trajectory states for the tracks at POCA for later cuts
      TrajectoryStateClosestToPoint posTSCP = 
	posTransTkPtr->trajectoryStateClosestToPoint( cxPt );
      TrajectoryStateClosestToPoint negTSCP =
	negTransTkPtr->trajectoryStateClosestToPoint( cxPt );

      if( !posTSCP.isValid() || !negTSCP.isValid() ) continue;


      /*double posESq = posTSCP.momentum().mag2() + piMassSquared;
      double negESq = negTSCP.momentum().mag2() + piMassSquared;
      double posE = sqrt(posESq);
      double negE = sqrt(negESq);
      double totalE = posE + negE;*/
      double totalE = sqrt( posTSCP.momentum().mag2() + piMassSquared ) +
	              sqrt( negTSCP.momentum().mag2() + piMassSquared );
      double totalESq = totalE*totalE;
      double totalPSq =
	( posTSCP.momentum() + negTSCP.momentum() ).mag2();
      double mass = sqrt( totalESq - totalPSq);

      //mPiPiMassOut << mass << std::endl;

      if( mass > mPiPiCut ) continue;

      // Create the vertex fitter object and vertex the tracks
      TransientVertex theRecoVertex;
      if(vtxFitter == std::string("KalmanVertexFitter")) {
	KalmanVertexFitter theKalmanFitter(useRefTrax == 0 ? false : true);
	theRecoVertex = theKalmanFitter.vertex(transTracks);
      }
      else if (vtxFitter == std::string("AdaptiveVertexFitter")) {
	useRefTrax = false;
	AdaptiveVertexFitter theAdaptiveFitter;
	theRecoVertex = theAdaptiveFitter.vertex(transTracks);
      }
    
      // If the vertex is valid, make a VertexCompositeCandidate with it

      if( !theRecoVertex.isValid() || theRecoVertex.totalChiSquared() < 0. ) {
	continue;
      }

      // Create reco::Vertex object for use in creating the Candidate
      reco::Vertex theVtx = theRecoVertex;
      // Create and fill vector of refitted TransientTracks
      //  (iff they've been created by the KVF)
      std::vector<TransientTrack> refittedTrax;
      if( theRecoVertex.hasRefittedTracks() ) {
	refittedTrax = theRecoVertex.refittedTracks();
      }

      // Do post-fit cuts if specified in config file.

      // Find the vertex d0 and its error

      typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3> > SMatrixSym3D;
      typedef ROOT::Math::SVector<double, 3> SVector3;

      GlobalPoint vtxPos(theVtx.x(), theVtx.y(), theVtx.z());

      GlobalPoint beamSpotPos(theBeamSpotHandle->position().x(),
			      theBeamSpotHandle->position().y(),
			      theBeamSpotHandle->position().z());

      SMatrixSym3D totalCov = theBeamSpotHandle->rotatedCovariance3D() + theVtx.covariance();
      SVector3 distanceVector(vtxPos.x() - beamSpotPos.x(),
			      vtxPos.y() - beamSpotPos.y(),
			      0.);//so that we get radial values only, 
                                  //since z beamSpot uncertainty is huge

      double rVtxMag = ROOT::Math::Mag(distanceVector);
      double sigmaRvtxMag = sqrt(ROOT::Math::Similarity(totalCov, distanceVector)) / rVtxMag;
      
      // The methods innerOk() and innerPosition() require TrackExtra, which
      // is only available in the RECO data tier, not AOD. Setting innerHitPosCut
      // to -1 avoids this problem and allows to run on AOD.
      if( innerHitPosCut > 0. && positiveTrackRef->innerOk() ) {
	reco::Vertex::Point posTkHitPos = positiveTrackRef->innerPosition();
	double posTkHitPosD2 = 
	  (posTkHitPos.x()-beamSpotPos.x())*(posTkHitPos.x()-beamSpotPos.x()) +
	  (posTkHitPos.y()-beamSpotPos.y())*(posTkHitPos.y()-beamSpotPos.y());
	if( sqrt( posTkHitPosD2 ) < ( rVtxMag - sigmaRvtxMag*innerHitPosCut )
	    ) {
	  continue;
	}
      }
      if( innerHitPosCut > 0. && negativeTrackRef->innerOk() ) {
	reco::Vertex::Point negTkHitPos = negativeTrackRef->innerPosition();
	double negTkHitPosD2 = 
	  (negTkHitPos.x()-beamSpotPos.x())*(negTkHitPos.x()-beamSpotPos.x()) +
	  (negTkHitPos.y()-beamSpotPos.y())*(negTkHitPos.y()-beamSpotPos.y());
	if( sqrt( negTkHitPosD2 ) < ( rVtxMag - sigmaRvtxMag*innerHitPosCut )
	    ) {
	  continue;
	}
      }
      
      if( theVtx.normalizedChi2() > chi2Cut ||
	  rVtxMag < rVtxCut ||
	  rVtxMag / sigmaRvtxMag < vtxSigCut ) {
	continue;
      }

      // Cuts finished, now we create the candidates and push them back into the collections.
      
      std::auto_ptr<TrajectoryStateClosestToPoint> trajPlus;
      std::auto_ptr<TrajectoryStateClosestToPoint> trajMins;

      if( useRefTrax && refittedTrax.size() > 1 ) {
	// Need an iterator over the refitted tracks for below
	std::vector<TransientTrack>::iterator traxIter = refittedTrax.begin(),
	  traxEnd = refittedTrax.end();

	// TransientTrack objects to hold the positive and negative
	//  refitted tracks
	TransientTrack* thePositiveRefTrack = 0;
	TransientTrack* theNegativeRefTrack = 0;
        
	for( ; traxIter != traxEnd; ++traxIter) {
	  if( traxIter->track().charge() > 0. ) {
	    thePositiveRefTrack = &*traxIter;
	  }
	  else if (traxIter->track().charge() < 0.) {
	    theNegativeRefTrack = &*traxIter;
	  }
	}
        if (thePositiveRefTrack == 0 || theNegativeRefTrack == 0) continue;
	trajPlus.reset(new TrajectoryStateClosestToPoint(thePositiveRefTrack->trajectoryStateClosestToPoint(vtxPos)));
	trajMins.reset(new TrajectoryStateClosestToPoint(theNegativeRefTrack->trajectoryStateClosestToPoint(vtxPos)));
      }
      else {
	trajPlus.reset(new TrajectoryStateClosestToPoint(posTransTkPtr->trajectoryStateClosestToPoint(vtxPos)));
	trajMins.reset(new TrajectoryStateClosestToPoint(negTransTkPtr->trajectoryStateClosestToPoint(vtxPos)));

      }

      if( trajPlus.get() == 0 || trajMins.get() == 0 || !trajPlus->isValid() || !trajMins->isValid() ) continue;

      posTransTkPtr = negTransTkPtr = 0;

      GlobalVector positiveP(trajPlus->momentum());
      GlobalVector negativeP(trajMins->momentum());
      GlobalVector totalP(positiveP + negativeP);

      //cleanup stuff we don't need anymore
      trajPlus.reset();
      trajMins.reset();

      // calculate total energy of V0 3 ways:
      //  Assume it's a kShort, a Lambda, or a LambdaBar.
      double piPlusE = sqrt( positiveP.mag2() + piMassSquared );
      double piMinusE = sqrt( negativeP.mag2() + piMassSquared );
      double protonE = sqrt( positiveP.mag2() + protonMassSquared );
      double antiProtonE = sqrt( negativeP.mag2() + protonMassSquared );
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
	if( theKshort->mass() < kShortMass + kShortMassCut &&
	    theKshort->mass() > kShortMass - kShortMassCut ) {
	  theKshorts.push_back( *theKshort );
	}
      }
      
      if( doLambdas && theLambda ) {
	theLambda->addDaughter(theProtonCand);
	theLambda->addDaughter(thePiMinusCand);
	theLambda->setPdgId(3122);
	addp4.set( *theLambda );
	if( theLambda->mass() < lambdaMass + lambdaMassCut &&
	    theLambda->mass() > lambdaMass - lambdaMassCut ) {
	  theLambdas.push_back( *theLambda );
	}
      }
      else if ( doLambdas && theLambdaBar ) {
	theLambdaBar->addDaughter(theAntiProtonCand);
	theLambdaBar->addDaughter(thePiPlusCand);
	theLambdaBar->setPdgId(-3122);
	addp4.set( *theLambdaBar );
	if( theLambdaBar->mass() < lambdaMass + lambdaMassCut &&
	    theLambdaBar->mass() > lambdaMass - lambdaMassCut ) {
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

// Get methods
const reco::VertexCompositeCandidateCollection& V0Fitter::getKshorts() const {
  return theKshorts;
}

const reco::VertexCompositeCandidateCollection& V0Fitter::getLambdas() const {
  return theLambdas;
}


// Experimental
double V0Fitter::findV0MassError(const GlobalPoint &vtxPos, const std::vector<reco::TransientTrack> &dauTracks) { 
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



