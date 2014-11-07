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
   const double kShortMass = 0.497614;
   const double lambdaMass = 1.115683;
}

typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3> > SMatrixSym3D;
typedef ROOT::Math::SVector<double, 3> SVector3;

V0Fitter::V0Fitter(const edm::ParameterSet& theParameters, edm::ConsumesCollector && iC)
{
   using std::string;

   token_beamSpot = iC.consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
   token_tracks = iC.consumes<reco::TrackCollection>(theParameters.getParameter<edm::InputTag>("trackRecoAlgorithm"));

   vertexFitter_ = theParameters.getParameter<bool>("vertexFitter");
   useRefTracks_ = theParameters.getParameter<bool>("useRefTracks");
   
   // whether to reconstruct KShorts
   doKShorts_ = theParameters.getParameter<bool>("doKShorts");
   // whether to reconstruct Lambdas
   doLambdas_ = theParameters.getParameter<bool>("doLambdas");

   // cuts on initial track selection
   tkChi2Cut_ = theParameters.getParameter<double>("tkChi2Cut");
   tkNHitsCut_ = theParameters.getParameter<int>("tkNHitsCut");
   tkPtCut_ = theParameters.getParameter<double>("tkPtCut");
   tkIPSigCut_ = theParameters.getParameter<double>("tkIPSigCut");
   // cuts on vertex
   vtxChi2Cut_ = theParameters.getParameter<double>("vtxChi2Cut");
   vtxDecayRSigCut_ = theParameters.getParameter<double>("vtxDecayRSigCut");
   // miscellaneous cuts
   tkDCACut_ = theParameters.getParameter<double>("tkDCACut");
   mPiPiCut_ = theParameters.getParameter<double>("mPiPiCut");
   innerHitPosCut_ = theParameters.getParameter<double>("innerHitPosCut");
   v0CosThetaCut_ = theParameters.getParameter<double>("v0CosThetaCut");
   // cuts on the V0 candidate mass
   kShortMassCut_ = theParameters.getParameter<double>("kShortMassCut");
   lambdaMassCut_ = theParameters.getParameter<double>("lambdaMassCut");
}

// method containing the algorithm for vertex reconstruction
void V0Fitter::fitAll(const edm::Event& iEvent, const edm::EventSetup& iSetup,
   reco::VertexCompositeCandidateCollection & theKshorts, reco::VertexCompositeCandidateCollection & theLambdas)
{
   using std::vector;
   using namespace reco;
   using namespace edm;

   Handle<reco::TrackCollection> theTrackHandle;
   iEvent.getByToken(token_tracks, theTrackHandle);
   if (!theTrackHandle->size()) return;
   const reco::TrackCollection* theTrackCollection = theTrackHandle.product();   

   Handle<reco::BeamSpot> theBeamSpotHandle;
   iEvent.getByToken(token_beamSpot, theBeamSpotHandle);
   const reco::BeamSpot* theBeamSpot = theBeamSpotHandle.product();
   const GlobalPoint beamSpotPos(theBeamSpot->position().x(), theBeamSpot->position().y(), theBeamSpot->position().z());

   ESHandle<MagneticField> theMagneticFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(theMagneticFieldHandle);
   const MagneticField* theMagneticField = theMagneticFieldHandle.product();

   std::vector<TrackRef> theTrackRefs;
   std::vector<TransientTrack> theTransTracks;

   // fill vectors of TransientTracks and TrackRefs after applying preselection cuts
   for (reco::TrackCollection::const_iterator iTk = theTrackCollection->begin(); iTk != theTrackCollection->end(); ++iTk) {
      const reco::Track* tmpTrack = &(*iTk);
      double ipsig = std::abs(tmpTrack->dxy(*theBeamSpot)/tmpTrack->dxyError());
      if (tmpTrack->normalizedChi2() < tkChi2Cut_ && tmpTrack->numberOfValidHits() >= tkNHitsCut_ &&
          tmpTrack->pt() > tkPtCut_ && ipsig > tkIPSigCut_) {
         TrackRef tmpRef(theTrackHandle, std::distance(theTrackCollection->begin(), iTk));
         theTrackRefs.push_back(std::move(tmpRef));
         TransientTrack tmpTransient(*tmpRef, theMagneticField);
         theTransTracks.push_back(std::move(tmpTransient));
      }
   }
   // good tracks have now been selected for vertexing

   // loop over tracks and vertex good charged track pairs
   for (unsigned int trdx1 = 0; trdx1 < theTrackRefs.size(); ++trdx1) {
   for (unsigned int trdx2 = trdx1 + 1; trdx2 < theTrackRefs.size(); ++trdx2) {

      TrackRef positiveTrackRef;
      TrackRef negativeTrackRef;
      TransientTrack* posTransTkPtr = nullptr;
      TransientTrack* negTransTkPtr = nullptr;

      if (theTrackRefs[trdx1]->charge() < 0. && theTrackRefs[trdx2]->charge() > 0.) {
         negativeTrackRef = theTrackRefs[trdx1];
         positiveTrackRef = theTrackRefs[trdx2];
         negTransTkPtr = &theTransTracks[trdx1];
         posTransTkPtr = &theTransTracks[trdx2];
      } else if (theTrackRefs[trdx1]->charge() > 0. && theTrackRefs[trdx2]->charge() < 0.) {
         negativeTrackRef = theTrackRefs[trdx2];
         positiveTrackRef = theTrackRefs[trdx1];
         negTransTkPtr = &theTransTracks[trdx2];
         posTransTkPtr = &theTransTracks[trdx1];
      } else {
         continue;
      }

      // measure distance between tracks at their closest approach
      if (!posTransTkPtr->impactPointTSCP().isValid() || !negTransTkPtr->impactPointTSCP().isValid()) continue;
      FreeTrajectoryState const & posState = posTransTkPtr->impactPointTSCP().theState();
      FreeTrajectoryState const & negState = negTransTkPtr->impactPointTSCP().theState();
      ClosestApproachInRPhi cApp;
      cApp.calculate(posState, negState);
      if (!cApp.status()) continue;
      float dca = std::abs(cApp.distance());
      if (dca > tkDCACut_) continue;

      // the POCA should at least be in the sensitive volume
      GlobalPoint cxPt = cApp.crossingPoint();
      if (sqrt(cxPt.x()*cxPt.x() + cxPt.y()*cxPt.y()) > 120. || std::abs(cxPt.z()) > 300.) continue;

      // the tracks should at least point in the same quadrant
      TrajectoryStateClosestToPoint const & posTSCP = posTransTkPtr->trajectoryStateClosestToPoint(cxPt);
      TrajectoryStateClosestToPoint const & negTSCP = negTransTkPtr->trajectoryStateClosestToPoint(cxPt);
      if (!posTSCP.isValid() || !negTSCP.isValid()) continue;
      if (posTSCP.momentum().dot(negTSCP.momentum())  < 0) continue;
     
      // calculate mPiPi
      double totalE = sqrt(posTSCP.momentum().mag2() + piMassSquared) + sqrt(negTSCP.momentum().mag2() + piMassSquared);
      double totalESq = totalE*totalE;
      double totalPSq = (posTSCP.momentum() + negTSCP.momentum()).mag2();
      double mass = sqrt(totalESq - totalPSq);
      if (mass > mPiPiCut_) continue;

      // Fill the vector of TransientTracks to send to KVF
      std::vector<TransientTrack> transTracks;
      transTracks.reserve(2);
      transTracks.push_back(*posTransTkPtr);
      transTracks.push_back(*negTransTkPtr);

      // Create the vertex fitter object and vertex the tracks
      TransientVertex theRecoVertex;
      if (vertexFitter_) {
         KalmanVertexFitter theKalmanFitter(useRefTracks_ == 0 ? false : true);
         theRecoVertex = theKalmanFitter.vertex(transTracks);
      } else if (!vertexFitter_) {
         useRefTracks_ = false;
         AdaptiveVertexFitter theAdaptiveFitter;
         theRecoVertex = theAdaptiveFitter.vertex(transTracks);
      }
      if (!theRecoVertex.isValid()) continue;
     
      reco::Vertex theVtx = theRecoVertex;
      if (theVtx.normalizedChi2() > vtxChi2Cut_) continue;

      GlobalPoint vtxPos(theVtx.x(), theVtx.y(), theVtx.z());
      SMatrixSym3D totalCov = theBeamSpot->rotatedCovariance3D() + theVtx.covariance();
      SVector3 distanceVector(vtxPos.x() - beamSpotPos.x(), vtxPos.y() - beamSpotPos.y(), 0.);
      double rVtxMag = ROOT::Math::Mag(distanceVector);
      double sigmaRvtxMag = sqrt(ROOT::Math::Similarity(totalCov, distanceVector)) / rVtxMag;
      if (rVtxMag/sigmaRvtxMag < vtxDecayRSigCut_) continue;

      if (innerHitPosCut_ > 0. && positiveTrackRef->innerOk()) {
         reco::Vertex::Point posTkHitPos = positiveTrackRef->innerPosition();
         double posTkHitPosD2 =  (posTkHitPos.x()-beamSpotPos.x())*(posTkHitPos.x()-beamSpotPos.x()) +
            (posTkHitPos.y()-beamSpotPos.y())*(posTkHitPos.y()-beamSpotPos.y());
         if (sqrt(posTkHitPosD2) < (rVtxMag - sigmaRvtxMag*innerHitPosCut_)) continue;
      }
      if (innerHitPosCut_ > 0. && negativeTrackRef->innerOk()) {
         reco::Vertex::Point negTkHitPos = negativeTrackRef->innerPosition();
         double negTkHitPosD2 = (negTkHitPos.x()-beamSpotPos.x())*(negTkHitPos.x()-beamSpotPos.x()) +
            (negTkHitPos.y()-beamSpotPos.y())*(negTkHitPos.y()-beamSpotPos.y());
         if (sqrt(negTkHitPosD2) < (rVtxMag - sigmaRvtxMag*innerHitPosCut_)) continue;
      }
      
      std::auto_ptr<TrajectoryStateClosestToPoint> trajPlus;
      std::auto_ptr<TrajectoryStateClosestToPoint> trajMins;
      std::vector<TransientTrack> theRefTracks;
      if (theRecoVertex.hasRefittedTracks()) {
         theRefTracks = theRecoVertex.refittedTracks();
      }

      if (useRefTracks_ && theRefTracks.size() > 1) {
         TransientTrack* thePositiveRefTrack = 0;
         TransientTrack* theNegativeRefTrack = 0;
         for (std::vector<TransientTrack>::iterator iTrack = theRefTracks.begin(); iTrack != theRefTracks.end(); ++iTrack) {
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
         trajPlus.reset(new TrajectoryStateClosestToPoint(posTransTkPtr->trajectoryStateClosestToPoint(vtxPos)));
         trajMins.reset(new TrajectoryStateClosestToPoint(negTransTkPtr->trajectoryStateClosestToPoint(vtxPos)));
      }

      if (trajPlus.get() == 0 || trajMins.get() == 0 || !trajPlus->isValid() || !trajMins->isValid()) continue;

      GlobalVector positiveP(trajPlus->momentum());
      GlobalVector negativeP(trajMins->momentum());
      GlobalVector totalP(positiveP + negativeP);

      // calculate the pointing angle
      double posx = theVtx.x() - beamSpotPos.x();
      double posy = theVtx.y() - beamSpotPos.y();
      double momx = totalP.x();
      double momy = totalP.y();
      double pointangle = (posx*momx+posy*momy)/(sqrt(posx*posx+posy*posy)*sqrt(momx*momx+momy*momy));
      if (pointangle < v0CosThetaCut_) continue;

      // calculate total energy of V0 3 ways: assume it's a kShort, a Lambda, or a LambdaBar.
      double piPlusE = sqrt(positiveP.mag2() + piMassSquared);
      double piMinusE = sqrt(negativeP.mag2() + piMassSquared);
      double protonE = sqrt(positiveP.mag2() + protonMassSquared);
      double antiProtonE = sqrt(negativeP.mag2() + protonMassSquared);
      double kShortETot = piPlusE + piMinusE;
      double lambdaEtot = protonE + piMinusE;
      double lambdaBarEtot = antiProtonE + piPlusE;

      // Create momentum 4-vectors for the 3 candidate types
      const Particle::LorentzVector kShortP4(totalP.x(), totalP.y(), totalP.z(), kShortETot);
      const Particle::LorentzVector lambdaP4(totalP.x(), totalP.y(), totalP.z(), lambdaEtot);
      const Particle::LorentzVector lambdaBarP4(totalP.x(), totalP.y(), totalP.z(), lambdaBarEtot);

      Particle::Point vtx(theVtx.x(), theVtx.y(), theVtx.z());
      const Vertex::CovarianceMatrix vtxCov(theVtx.covariance());
      double vtxChi2(theVtx.chi2());
      double vtxNdof(theVtx.ndof());

      // Create the VertexCompositeCandidate object that will be stored in the Event
      VertexCompositeCandidate* theKshort = nullptr;
      VertexCompositeCandidate* theLambda = nullptr;
      VertexCompositeCandidate* theLambdaBar = nullptr;

      if (doKShorts_) {
         theKshort = new VertexCompositeCandidate(0, kShortP4, vtx, vtxCov, vtxChi2, vtxNdof);
      }
      if (doLambdas_) {
         if (positiveP.mag2() > negativeP.mag2()) {
            theLambda = new VertexCompositeCandidate(0, lambdaP4, vtx, vtxCov, vtxChi2, vtxNdof);
         } else {
            theLambdaBar = new VertexCompositeCandidate(0, lambdaBarP4, vtx, vtxCov, vtxChi2, vtxNdof);
         }
      }

      // Create daughter candidates for the VertexCompositeCandidates
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
      // Store the daughter Candidates in the VertexCompositeCandidates if they pass mass cuts
      if (doKShorts_) {
         theKshort->addDaughter(thePiPlusCand);
         theKshort->addDaughter(thePiMinusCand);
         theKshort->setPdgId(310);
         addp4.set(*theKshort);
         if (theKshort->mass() < kShortMass + kShortMassCut_ && theKshort->mass() > kShortMass - kShortMassCut_) {
            theKshorts.push_back(std::move(*theKshort));
         }
      }
      if (doLambdas_ && theLambda) {
         theLambda->addDaughter(theProtonCand);
         theLambda->addDaughter(thePiMinusCand);
         theLambda->setPdgId(3122);
         addp4.set( *theLambda );
         if (theLambda->mass() < lambdaMass + lambdaMassCut_ && theLambda->mass() > lambdaMass - lambdaMassCut_) {
            theLambdas.push_back(std::move(*theLambda));
         }
      } else if (doLambdas_ && theLambdaBar) {
         theLambdaBar->addDaughter(theAntiProtonCand);
         theLambdaBar->addDaughter(thePiPlusCand);
         theLambdaBar->setPdgId(-3122);
         addp4.set(*theLambdaBar);
         if (theLambdaBar->mass() < lambdaMass + lambdaMassCut_ && theLambdaBar->mass() > lambdaMass - lambdaMassCut_) {
            theLambdas.push_back(std::move(*theLambdaBar));
         }
      }

      delete theKshort;
      delete theLambda;
      delete theLambdaBar;
      theKshort = theLambda = theLambdaBar = nullptr;

    }
  }
}

