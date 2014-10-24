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
//
//

#include "V0Fitter.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <typeinfo>
#include <memory>

// pdg mass constants
namespace {
   const double piMass = 0.13957018;
   const double piMassSquared = piMass*piMass;
   const double protonMass = 0.938272046;
   const double protonMassSquared = protonMass*protonMass;
   const double kshortMass = 0.497614;
   const double lambdaMass = 1.115683;
}

// constructor
V0Fitter::V0Fitter(const edm::ParameterSet& theParameters, edm::ConsumesCollector && iC)
{
   using std::string;

   // token to request the tracks (with a parameter to decide which track collection)
   token_tracks = iC.consumes<reco::TrackCollection>(theParameters.getParameter<edm::InputTag>("trackRecoAlgorithm"));

   // token to request the beamspot
   token_beamspot = iC.consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));

   // whether to reconstruct K0s
   doKShorts_ = theParameters.getParameter<bool>(string("doKShorts"));
   // whether to reconstruct Lambdas
   doLambdas_ = theParameters.getParameter<bool>(string("doLambdas"));

   // which vertex fitting algorithm to use
   vtxFitter = theParameters.getParameter<edm::InputTag>("vertexFitter");

   // whether to use the refit tracks for V0 kinematics
   useRefTrax = theParameters.getParameter<bool>(string("useSmoothing"));

   // cuts on the input track collection
   std::vector<std::string> qual = theParameters.getParameter<std::vector<std::string> >("trackQualities");
   for (unsigned int ndx = 0; ndx < qual.size(); ndx++) {
      qualities.push_back(reco::TrackBase::qualityByName(qual[ndx]));
   }
   tkChi2Cut_ = theParameters.getParameter<double>(string("tkChi2Cut"));
   tkPtCut_ = theParameters.getParameter<double>(string("tkPtCut"));  
   tkNHitsCut_ = theParameters.getParameter<int>(string("tkNHitsCut"));
   tkIPSigCut_ = theParameters.getParameter<double>(string("tkIPSigCut"));
   // cuts on the V0 vertex
   vtxChi2Cut_ = theParameters.getParameter<double>(string("vtxChi2Cut"));
   vtxRCut_ = theParameters.getParameter<double>(string("vtxRCut"));
   vtxRSigCut_ = theParameters.getParameter<double>(string("vtxRSigCut"));
   // miscellaneous cuts after vertexing
   tkDCACut_ = theParameters.getParameter<double>(string("tkDCACut"));
   kshortMassCut_ = theParameters.getParameter<double>(string("kshortMassCut"));
   lambdaMassCut_ = theParameters.getParameter<double>(string("lambdaMassCut"));
   innerHitPosCut_ = theParameters.getParameter<double>(string("innerHitPosCut"));
   v0CosThetaCut_ = theParameters.getParameter<double>(string("v0CosThetaCut"));

}

// method containing the algorithm for vertex reconstruction
void V0Fitter::fitAll(const edm::Event& iEvent, const edm::EventSetup& iSetup,
   reco::VertexCompositeCandidateCollection & theKShorts, reco::VertexCompositeCandidateCollection & theLambdas)
{
   using std::vector;
   using namespace reco;
   using namespace edm;

   edm::Handle<reco::TrackCollection> trackHandle;
   iEvent.getByToken(token_tracks, trackHandle);
   if (!trackHandle->size()) return;
   const reco::TrackCollection* trackColl = trackHandle.product();
  
   edm::Handle<reco::BeamSpot> beamspotHandle;
   iEvent.getByToken(token_beamspot, beamspotHandle);
   const GlobalPoint beamspotPos(beamspotHandle->position().x(), beamspotHandle->position().y(), beamspotHandle->position().z());

   ESHandle<MagneticField> bFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
   const MagneticField* magField = bFieldHandle.product();

   ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
   iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

   // fill theGoodTracks with track index if it passes preselection
   std::vector<size_t> theGoodTracks;
   for (TrackCollection::const_iterator iTrack = trackColl->begin(); iTrack != trackColl->end(); ++iTrack) {
      bool quality_ok = true;
      if (qualities.size() != 0) {
         quality_ok = false;
         for (unsigned int ndx_ = 0; ndx_ < qualities.size(); ++ndx_) {
            if ((*iTrack).quality(qualities[ndx_])) {
               quality_ok = true;
               break;
            }
         }
      }
      if (quality_ok) {
         if ((*iTrack).normalizedChi2() < tkChi2Cut_ && (*iTrack).pt() > tkPtCut_ && (*iTrack).numberOfValidHits() >= tkNHitsCut_) {
            FreeTrajectoryState initialFTS = trajectoryStateTransform::initialFreeState((*iTrack), magField);
            TSCBLBuilderNoMaterial blsBuilder;
            TrajectoryStateClosestToBeamLine tscb(blsBuilder(initialFTS, *beamspotHandle));
            if (tscb.isValid()) {
               if (tscb.transverseImpactParameter().significance() > tkIPSigCut_) {
                  theGoodTracks.push_back(std::move(std::distance(trackColl->begin(), iTrack)));
               }
            }
         }
      }
   }
   // good tracks have now been selected for vertexing

   // loop over the tracks and vertex good charged track pairs
   for (unsigned int trdx1 = 0; trdx1 < theGoodTracks.size(); ++trdx1) {
   for (unsigned int trdx2 = trdx1 + 1; trdx2 < theGoodTracks.size(); ++trdx2) {

      TrackRef negativeTrackRef;
      TrackRef positiveTrackRef;
      TrackRef temp1(trackHandle, theGoodTracks[trdx1]);
      TrackRef temp2(trackHandle, theGoodTracks[trdx2]);
       
      // if the tracks are oppositely charged load them into the appropriate containers
      if (temp1->charge() < 0. && temp2->charge() > 0.) {
         negativeTrackRef = temp1;
         positiveTrackRef = temp2;
      } else if (temp1->charge() > 0. && temp2->charge() < 0.) {
         negativeTrackRef = temp2;
         positiveTrackRef = temp1;
      } else {
         continue; // try the next pair
      }

      TransientTrack negTransTk(*negativeTrackRef, &(*bFieldHandle), globTkGeomHandle);
      TransientTrack posTransTk(*positiveTrackRef, &(*bFieldHandle), globTkGeomHandle);
     
      // calculate the DCA and POCA for the track pair
      if (!negTransTk.impactPointTSCP().isValid() || !posTransTk.impactPointTSCP().isValid()) continue;
      FreeTrajectoryState const & negState = negTransTk.impactPointTSCP().theState();
      FreeTrajectoryState const & posState = posTransTk.impactPointTSCP().theState();
      ClosestApproachInRPhi cApp;
      cApp.calculate(posState, negState);
      if (!cApp.status()) continue;
      float dca = fabs(cApp.distance());
      if (dca > tkDCACut_) continue;
      GlobalPoint cxPt = cApp.crossingPoint();
      if (sqrt(cxPt.x()*cxPt.x() + cxPt.y()*cxPt.y()) > 120. || std::abs(cxPt.z()) > 300.) continue;

      // fill the vector of TransientTracks to give to the vertexers
      std::vector<TransientTrack> transTracks;
      transTracks.reserve(2);
      transTracks.push_back(negTransTk);
      transTracks.push_back(posTransTk);

      // create the vertex fitter object and vertex the tracks
      TransientVertex theRecoVertex;
      if (vtxFitter == std::string("KalmanVertexFitter")) {
         KalmanVertexFitter theKalmanFitter(useRefTrax == 0 ? false : true);
         theRecoVertex = theKalmanFitter.vertex(transTracks);
      } else if (vtxFitter == std::string("AdaptiveVertexFitter")) {
         useRefTrax = false;
         AdaptiveVertexFitter theAdaptiveFitter;
         theRecoVertex = theAdaptiveFitter.vertex(transTracks);
      }
      if (!theRecoVertex.isValid()) continue;

      // create reco::Vertex object for use in creating the Candidate
      reco::Vertex theVtx = theRecoVertex;
      if (theVtx.normalizedChi2() > vtxChi2Cut_) continue;

      // calculate radial displacement of V0 vertex from beamspot (z uncertainty is large)
      typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3> > SMatrixSym3D;
      typedef ROOT::Math::SVector<double, 3> SVector3;
      SMatrixSym3D totalCov = beamspotHandle->rotatedCovariance3D() + theVtx.covariance();
      SVector3 distanceVector(theVtx.x()-beamspotPos.x(), theVtx.y()-beamspotPos.y(), 0.);
      double rVtxMag = ROOT::Math::Mag(distanceVector);
      if (rVtxMag < vtxRCut_) continue;
      double sigmaRvtxMag = sqrt(ROOT::Math::Similarity(totalCov, distanceVector)) / rVtxMag;
      if (rVtxMag / sigmaRvtxMag < vtxRSigCut_) continue;

      // see if either daughter track has hits "inside" the vertex
      // (the methods innerOk() and innerPosition() require TrackExtra which is only available in the RECO data tier - 
      //  setting innerHitPosCut to -1 avoids this problem and allows to run on AOD)
      if (innerHitPosCut_ > 0. && positiveTrackRef->innerOk()) {
         reco::Vertex::Point posTkHitPos = positiveTrackRef->innerPosition();
         double posTkHitPosD2 =
            (posTkHitPos.x()-beamspotPos.x())*(posTkHitPos.x()-beamspotPos.x()) +
            (posTkHitPos.y()-beamspotPos.y())*(posTkHitPos.y()-beamspotPos.y());
         if (sqrt(posTkHitPosD2) < (rVtxMag - sigmaRvtxMag*innerHitPosCut_)) continue;
      }
      if (innerHitPosCut_ > 0. && negativeTrackRef->innerOk()) {
         reco::Vertex::Point negTkHitPos = negativeTrackRef->innerPosition();
         double negTkHitPosD2 =
            (negTkHitPos.x()-beamspotPos.x())*(negTkHitPos.x()-beamspotPos.x()) +
            (negTkHitPos.y()-beamspotPos.y())*(negTkHitPos.y()-beamspotPos.y());
         if (sqrt(negTkHitPosD2) < (rVtxMag - sigmaRvtxMag*innerHitPosCut_)) continue;
      }

      // create and fill vector of refitted TransientTracks (iff they've been created by the KVF)
      std::vector<TransientTrack> refittedTrax;
      if (theRecoVertex.hasRefittedTracks()) {
         refittedTrax = theRecoVertex.refittedTracks();
      }

      // make TrajectoryStates to extract momentum of daughter tracks at vertex
      std::auto_ptr<TrajectoryStateClosestToPoint> trajPlus;
      std::auto_ptr<TrajectoryStateClosestToPoint> trajMins;
      const GlobalPoint vtxPos(theVtx.x(), theVtx.y(), theVtx.z());

      if (useRefTrax && refittedTrax.size() > 1) {
         // TransientTrack objects to hold the positive and negative refitted tracks
         TransientTrack* thePositiveRefTrack = 0;
         TransientTrack* theNegativeRefTrack = 0;
         for (std::vector<TransientTrack>::iterator iTrack = refittedTrax.begin(); iTrack != refittedTrax.end(); ++iTrack) {
            if (iTrack->track().charge() > 0.) {
               thePositiveRefTrack = &*iTrack;
            } else if (iTrack->track().charge() < 0.) {
               theNegativeRefTrack = &*iTrack;
            }
         }
        if (thePositiveRefTrack == 0 || theNegativeRefTrack == 0) continue;
        trajPlus.reset(new TrajectoryStateClosestToPoint(thePositiveRefTrack->trajectoryStateClosestToPoint(vtxPos)));
        trajMins.reset(new TrajectoryStateClosestToPoint(theNegativeRefTrack->trajectoryStateClosestToPoint(vtxPos)));
      } else {
         trajPlus.reset(new TrajectoryStateClosestToPoint(posTransTk.trajectoryStateClosestToPoint(vtxPos)));
         trajMins.reset(new TrajectoryStateClosestToPoint(negTransTk.trajectoryStateClosestToPoint(vtxPos)));
      }
      if (trajPlus.get() == 0 || trajMins.get() == 0 || !trajPlus->isValid() || !trajMins->isValid()) continue;

      GlobalVector positiveP(trajPlus->momentum());
      GlobalVector negativeP(trajMins->momentum());
      GlobalVector totalP(positiveP + negativeP);

      // calculate the pointing angle
      double posx = theVtx.x() - beamspotHandle->position().x();
      double posy = theVtx.y() - beamspotHandle->position().y();
      double momx = totalP.x();
      double momy = totalP.y();
      double pointangle = (posx*momx+posy*momy)/(sqrt(posx*posx+posy*posy)*sqrt(momx*momx+momy*momy));
      if (pointangle < v0CosThetaCut_) continue;

      // calculate total energy of V0 3 ways: assume a KShort, Lambda, or LambdaBar
      double piPlusE = sqrt(positiveP.mag2() + piMassSquared);
      double piMinusE = sqrt(negativeP.mag2() + piMassSquared);
      double protonE = sqrt(positiveP.mag2() + protonMassSquared);
      double antiProtonE = sqrt(negativeP.mag2() + protonMassSquared);
      double kshortETot = piPlusE + piMinusE;
      double lambdaEtot = protonE + piMinusE;
      double lambdaBarEtot = antiProtonE + piPlusE;

      // create momentum 4-vectors for the 3 candidate types
      const Particle::LorentzVector kshortP4(totalP.x(), totalP.y(), totalP.z(), kshortETot);
      const Particle::LorentzVector lambdaP4(totalP.x(), totalP.y(), totalP.z(), lambdaEtot);
      const Particle::LorentzVector lambdaBarP4(totalP.x(), totalP.y(), totalP.z(), lambdaBarEtot);

      Particle::Point vtx(theVtx.x(), theVtx.y(), theVtx.z());
      const Vertex::CovarianceMatrix vtxCov(theVtx.covariance());
      double vtxChi2(theVtx.chi2());
      double vtxNdof(theVtx.ndof());

      // create the VertexCompositeCandidate object that will be stored in the Event
      VertexCompositeCandidate* theKShort = nullptr;
      VertexCompositeCandidate* theLambda = nullptr;
      VertexCompositeCandidate* theLambdaBar = nullptr;

      if (doKShorts_) {
         theKShort = new VertexCompositeCandidate(0, kshortP4, vtx, vtxCov, vtxChi2, vtxNdof);
      }
      if (doLambdas_) {
         if (positiveP.mag2() > negativeP.mag2()) {
            theLambda = new VertexCompositeCandidate(0, lambdaP4, vtx, vtxCov, vtxChi2, vtxNdof);
         } else {
            theLambdaBar = new VertexCompositeCandidate(0, lambdaBarP4, vtx, vtxCov, vtxChi2, vtxNdof);
         }
      }

      // create daughter candidates for the VertexCompositeCandidates
      RecoChargedCandidate thePiPlusCand(
         1, Particle::LorentzVector(positiveP.x(), positiveP.y(), positiveP.z(), piPlusE), vtx);
      thePiPlusCand.setTrack(positiveTrackRef);
      
      RecoChargedCandidate thePiMinusCand(
         -1, Particle::LorentzVector(negativeP.x(), negativeP.y(), negativeP.z(), piMinusE), vtx);
      thePiMinusCand.setTrack(negativeTrackRef);
      
      RecoChargedCandidate theProtonCand(
         1, Particle::LorentzVector(positiveP.x(), positiveP.y(), positiveP.z(), protonE), vtx);
      theProtonCand.setTrack(positiveTrackRef);

      RecoChargedCandidate theAntiProtonCand(
         -1, Particle::LorentzVector(negativeP.x(), negativeP.y(), negativeP.z(), antiProtonE), vtx);
      theAntiProtonCand.setTrack(negativeTrackRef);

      AddFourMomenta addp4;
      // store the daughter Candidates in the VertexCompositeCandidates if they pass mass cuts
      if (doKShorts_) {
         theKShort->addDaughter(thePiPlusCand);
         theKShort->addDaughter(thePiMinusCand);
         theKShort->setPdgId(310);
         addp4.set(*theKShort);
         if (theKShort->mass() < kshortMass+kshortMassCut_ && theKShort->mass() > kshortMass-kshortMassCut_) {
            theKShorts.push_back(std::move(*theKShort));
         }
      }      
      if (doLambdas_ && theLambda) {
         theLambda->addDaughter(theProtonCand);
         theLambda->addDaughter(thePiMinusCand);
         theLambda->setPdgId(3122);
         addp4.set(*theLambda);
         if (theLambda->mass() < lambdaMass+lambdaMassCut_ && theLambda->mass() > lambdaMass-lambdaMassCut_) {
            theLambdas.push_back(std::move(*theLambda));
         }
      } else if (doLambdas_ && theLambdaBar) {
         theLambdaBar->addDaughter(theAntiProtonCand);
         theLambdaBar->addDaughter(thePiPlusCand);
         theLambdaBar->setPdgId(-3122);
         addp4.set(*theLambdaBar);
         if (theLambdaBar->mass() < lambdaMass+lambdaMassCut_ && theLambdaBar->mass() > lambdaMass-lambdaMassCut_) {
            theLambdas.push_back(std::move(*theLambdaBar));
         }
      }

      delete theKShort;
      delete theLambda;
      delete theLambdaBar;
      theKShort = theLambda = theLambdaBar = nullptr;

   }
   }

}

