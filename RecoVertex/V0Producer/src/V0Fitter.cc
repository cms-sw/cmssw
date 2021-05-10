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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include <Math/Functions.h>
#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <memory>
#include <typeinfo>

// pdg mass constants
namespace {
  const double piMass = 0.13957018;
  const double piMassSquared = piMass * piMass;
  const double protonMass = 0.938272046;
  const double protonMassSquared = protonMass * protonMass;
  const double kShortMass = 0.497614;
  const double lambdaMass = 1.115683;
}  // namespace

typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>> SMatrixSym3D;
typedef ROOT::Math::SVector<double, 3> SVector3;

V0Fitter::V0Fitter(const edm::ParameterSet& theParameters, edm::ConsumesCollector&& iC) : esTokenMF_(iC.esConsumes()) {
  token_beamSpot = iC.consumes<reco::BeamSpot>(theParameters.getParameter<edm::InputTag>("beamSpot"));
  useVertex_ = theParameters.getParameter<bool>("useVertex");
  if (useVertex_)
    token_vertices = iC.consumes<std::vector<reco::Vertex>>(theParameters.getParameter<edm::InputTag>("vertices"));

  token_tracks = iC.consumes<reco::TrackCollection>(theParameters.getParameter<edm::InputTag>("trackRecoAlgorithm"));
  doFit_ = theParameters.getParameter<bool>("doFit");
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
  tkIPSigXYCut_ = theParameters.getParameter<double>("tkIPSigXYCut");
  tkIPSigZCut_ = theParameters.getParameter<double>("tkIPSigZCut");

  // cuts on vertex
  vtxChi2Cut_ = theParameters.getParameter<double>("vtxChi2Cut");
  vtxDecaySigXYZCut_ = theParameters.getParameter<double>("vtxDecaySigXYZCut");
  vtxDecaySigXYCut_ = theParameters.getParameter<double>("vtxDecaySigXYCut");
  vtxDecayXYCut_ = theParameters.getParameter<double>("vtxDecayXYCut");
  ssVtxDecayXYCut_ = theParameters.getParameter<double>("ssVtxDecayXYCut");
  // miscellaneous cuts
  allowSS_ = theParameters.getParameter<bool>("allowSS");
  innerTkDCACut_ = theParameters.getParameter<double>("innerTkDCACut");
  outerTkDCACut_ = theParameters.getParameter<double>("outerTkDCACut");
  allowWideAngleVtx_ = theParameters.getParameter<bool>("allowWideAngleVtx");
  mPiPiCut_ = theParameters.getParameter<double>("mPiPiCut");
  innerHitPosCut_ = theParameters.getParameter<double>("innerHitPosCut");
  cosThetaXYCut_ = theParameters.getParameter<double>("cosThetaXYCut");
  cosThetaXYZCut_ = theParameters.getParameter<double>("cosThetaXYZCut");
  // cuts on the V0 candidate mass
  kShortMassCut_ = theParameters.getParameter<double>("kShortMassCut");
  lambdaMassCut_ = theParameters.getParameter<double>("lambdaMassCut");
}

// method containing the algorithm for vertex reconstruction
void V0Fitter::fitAll(const edm::Event& iEvent,
                      const edm::EventSetup& iSetup,
                      reco::VertexCompositeCandidateCollection& theKshorts,
                      reco::VertexCompositeCandidateCollection& theLambdas) {
  using std::vector;

  edm::Handle<reco::TrackCollection> theTrackHandle;
  iEvent.getByToken(token_tracks, theTrackHandle);
  if (theTrackHandle->empty())
    return;
  const reco::TrackCollection* theTrackCollection = theTrackHandle.product();

  edm::Handle<reco::BeamSpot> theBeamSpotHandle;
  iEvent.getByToken(token_beamSpot, theBeamSpotHandle);
  const reco::BeamSpot* theBeamSpot = theBeamSpotHandle.product();
  math::XYZPoint referencePos(theBeamSpot->position());

  reco::Vertex referenceVtx;
  if (useVertex_) {
    edm::Handle<std::vector<reco::Vertex>> vertices;
    iEvent.getByToken(token_vertices, vertices);
    referenceVtx = vertices->at(0);
    referencePos = referenceVtx.position();
  }

  const MagneticField* theMagneticField = &iSetup.getData(esTokenMF_);

  std::vector<reco::TrackRef> theTrackRefs;
  std::vector<reco::TransientTrack> theTransTracks;

  // fill vectors of TransientTracks and TrackRefs after applying preselection cuts
  for (reco::TrackCollection::const_iterator iTk = theTrackCollection->begin(); iTk != theTrackCollection->end();
       ++iTk) {
    const reco::Track* tmpTrack = &(*iTk);
    double ipsigXY = std::abs(tmpTrack->dxy(*theBeamSpot) / tmpTrack->dxyError());
    if (useVertex_)
      ipsigXY = std::abs(tmpTrack->dxy(referencePos) / tmpTrack->dxyError());
    double ipsigZ = std::abs(tmpTrack->dz(referencePos) / tmpTrack->dzError());
    if (tmpTrack->normalizedChi2() < tkChi2Cut_ && tmpTrack->numberOfValidHits() >= tkNHitsCut_ &&
        tmpTrack->pt() > tkPtCut_ && ipsigXY > tkIPSigXYCut_ && ipsigZ > tkIPSigZCut_) {
      reco::TrackRef tmpRef(theTrackHandle, std::distance(theTrackCollection->begin(), iTk));
      theTrackRefs.push_back(std::move(tmpRef));
      reco::TransientTrack tmpTransient(*tmpRef, theMagneticField);
      theTransTracks.push_back(std::move(tmpTransient));
    }
  }
  // good tracks have now been selected for vertexing

  // loop over tracks and vertex good charged track pairs
  for (unsigned int trdx1 = 0; trdx1 < theTrackRefs.size(); ++trdx1) {
    for (unsigned int trdx2 = trdx1 + 1; trdx2 < theTrackRefs.size(); ++trdx2) {
      reco::TrackRef positiveTrackRef;
      reco::TrackRef negativeTrackRef;
      reco::TransientTrack* posTransTkPtr = nullptr;
      reco::TransientTrack* negTransTkPtr = nullptr;

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
        if (!allowSS_)
          continue;

        // if same-sign pairs are allowed, assign the negative and positive tracks arbitrarily
        negativeTrackRef = theTrackRefs[trdx1];
        positiveTrackRef = theTrackRefs[trdx2];
        negTransTkPtr = &theTransTracks[trdx1];
        posTransTkPtr = &theTransTracks[trdx2];
      }

      // measure distance between tracks at their closest approach

      //these two variables are needed to 'pin' the temporary value returned to the stack
      // in order to keep posState and negState from pointing to destructed objects
      auto const& posImpact = posTransTkPtr->impactPointTSCP();
      auto const& negImpact = negTransTkPtr->impactPointTSCP();
      if (!posImpact.isValid() || !negImpact.isValid())
        continue;
      FreeTrajectoryState const& posState = posImpact.theState();
      FreeTrajectoryState const& negState = negImpact.theState();
      ClosestApproachInRPhi cApp;
      cApp.calculate(posState, negState);
      if (!cApp.status())
        continue;
      float dca = std::abs(cApp.distance());

      // the POCA should at least be in the sensitive volume
      GlobalPoint cxPt = cApp.crossingPoint();
      if ((cxPt.x() * cxPt.x() + cxPt.y() * cxPt.y()) > 120. * 120. || std::abs(cxPt.z()) > 300.)
        continue;

      if (cxPt.x() * cxPt.x() + cxPt.y() * cxPt.y() < 25.0) {
        if (dca > innerTkDCACut_)
          continue;
      } else {
        if (dca > outerTkDCACut_)
          continue;
      }

      // the tracks should at least point in the same quadrant
      TrajectoryStateClosestToPoint const& posTSCP = posTransTkPtr->trajectoryStateClosestToPoint(cxPt);
      TrajectoryStateClosestToPoint const& negTSCP = negTransTkPtr->trajectoryStateClosestToPoint(cxPt);
      if (!posTSCP.isValid() || !negTSCP.isValid())
        continue;
      if (!allowWideAngleVtx_ && posTSCP.momentum().dot(negTSCP.momentum()) < 0)
        continue;

      // calculate mPiPi
      double totalE = sqrt(posTSCP.momentum().mag2() + piMassSquared) + sqrt(negTSCP.momentum().mag2() + piMassSquared);
      double totalESq = totalE * totalE;
      double totalPSq = (posTSCP.momentum() + negTSCP.momentum()).mag2();
      double massSquared = totalESq - totalPSq;
      if (massSquared > mPiPiCut_ * mPiPiCut_)
        continue;

      // Fill the vector of TransientTracks to send to KVF
      std::vector<reco::TransientTrack> transTracks;
      transTracks.reserve(2);
      transTracks.push_back(*posTransTkPtr);
      transTracks.push_back(*negTransTkPtr);

      // create the vertex fitter object and vertex the tracks
      const GlobalError dummyError(1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 1.0e-3);
      TransientVertex theRecoVertex(cxPt, dummyError, transTracks, 1.0e-3);
      if (doFit_) {
        if (vertexFitter_) {
          KalmanVertexFitter theKalmanFitter(useRefTracks_ == 0 ? false : true);
          theRecoVertex = theKalmanFitter.vertex(transTracks);
        } else if (!vertexFitter_) {
          useRefTracks_ = false;
          AdaptiveVertexFitter theAdaptiveFitter;
          theRecoVertex = theAdaptiveFitter.vertex(transTracks);
        }
      }
      if (!theRecoVertex.isValid())
        continue;

      reco::Vertex theVtx = theRecoVertex;
      if (theVtx.normalizedChi2() > vtxChi2Cut_)
        continue;
      GlobalPoint vtxPos(theVtx.x(), theVtx.y(), theVtx.z());

      // 2D decay significance
      SMatrixSym3D totalCov = theBeamSpot->rotatedCovariance3D() + theVtx.covariance();
      if (useVertex_)
        totalCov = referenceVtx.covariance() + theVtx.covariance();
      SVector3 distVecXY(vtxPos.x() - referencePos.x(), vtxPos.y() - referencePos.y(), 0.);
      double distMagXY = ROOT::Math::Mag(distVecXY);
      double sigmaDistMagXY = sqrt(ROOT::Math::Similarity(totalCov, distVecXY)) / distMagXY;
      if (distMagXY / sigmaDistMagXY < vtxDecaySigXYCut_)
        continue;
      if (distMagXY < vtxDecayXYCut_)
        continue;
      if (posTransTkPtr->charge() * negTransTkPtr->charge() > 0 && distMagXY < ssVtxDecayXYCut_)
        continue;

      // 3D decay significance
      if (vtxDecaySigXYZCut_ > 0.) {
        SVector3 distVecXYZ(
            vtxPos.x() - referencePos.x(), vtxPos.y() - referencePos.y(), vtxPos.z() - referencePos.z());
        double distMagXYZ = ROOT::Math::Mag(distVecXYZ);
        double sigmaDistMagXYZ = sqrt(ROOT::Math::Similarity(totalCov, distVecXYZ)) / distMagXYZ;
        if (distMagXYZ / sigmaDistMagXYZ < vtxDecaySigXYZCut_)
          continue;
      }

      // make sure the vertex radius is within the inner track hit radius
      double tkHitPosLimitSquared =
          (distMagXY - sigmaDistMagXY * innerHitPosCut_) * (distMagXY - sigmaDistMagXY * innerHitPosCut_);
      if (innerHitPosCut_ > 0. && positiveTrackRef->innerOk()) {
        reco::Vertex::Point posTkHitPos = positiveTrackRef->innerPosition();
        double posTkHitPosD2 = (posTkHitPos.x() - referencePos.x()) * (posTkHitPos.x() - referencePos.x()) +
                               (posTkHitPos.y() - referencePos.y()) * (posTkHitPos.y() - referencePos.y());
        if (posTkHitPosD2 < tkHitPosLimitSquared)
          continue;
      }
      if (innerHitPosCut_ > 0. && negativeTrackRef->innerOk()) {
        reco::Vertex::Point negTkHitPos = negativeTrackRef->innerPosition();
        double negTkHitPosD2 = (negTkHitPos.x() - referencePos.x()) * (negTkHitPos.x() - referencePos.x()) +
                               (negTkHitPos.y() - referencePos.y()) * (negTkHitPos.y() - referencePos.y());
        if (negTkHitPosD2 < tkHitPosLimitSquared)
          continue;
      }

      std::unique_ptr<TrajectoryStateClosestToPoint> trajPlus;
      std::unique_ptr<TrajectoryStateClosestToPoint> trajMins;
      std::vector<reco::TransientTrack> theRefTracks;
      if (theRecoVertex.hasRefittedTracks()) {
        theRefTracks = theRecoVertex.refittedTracks();
      }

      if (useRefTracks_ && theRefTracks.size() > 1) {
        reco::TransientTrack* thePositiveRefTrack = nullptr;
        reco::TransientTrack* theNegativeRefTrack = nullptr;
        for (std::vector<reco::TransientTrack>::iterator iTrack = theRefTracks.begin(); iTrack != theRefTracks.end();
             ++iTrack) {
          if (iTrack->track().charge() > 0.) {
            thePositiveRefTrack = &*iTrack;
          } else if (iTrack->track().charge() < 0.) {
            theNegativeRefTrack = &*iTrack;
          }
        }
        if (thePositiveRefTrack == nullptr || theNegativeRefTrack == nullptr)
          continue;
        trajPlus =
            std::make_unique<TrajectoryStateClosestToPoint>(thePositiveRefTrack->trajectoryStateClosestToPoint(vtxPos));
        trajMins =
            std::make_unique<TrajectoryStateClosestToPoint>(theNegativeRefTrack->trajectoryStateClosestToPoint(vtxPos));
      } else {
        trajPlus =
            std::make_unique<TrajectoryStateClosestToPoint>(posTransTkPtr->trajectoryStateClosestToPoint(vtxPos));
        trajMins =
            std::make_unique<TrajectoryStateClosestToPoint>(negTransTkPtr->trajectoryStateClosestToPoint(vtxPos));
      }

      if (trajPlus.get() == nullptr || trajMins.get() == nullptr || !trajPlus->isValid() || !trajMins->isValid())
        continue;

      GlobalVector positiveP(trajPlus->momentum());
      GlobalVector negativeP(trajMins->momentum());
      GlobalVector totalP(positiveP + negativeP);

      // 2D pointing angle
      double dx = theVtx.x() - referencePos.x();
      double dy = theVtx.y() - referencePos.y();
      double px = totalP.x();
      double py = totalP.y();
      double angleXY = (dx * px + dy * py) / (sqrt(dx * dx + dy * dy) * sqrt(px * px + py * py));
      if (angleXY < cosThetaXYCut_)
        continue;

      // 3D pointing angle
      if (cosThetaXYZCut_ > -1.) {
        double dz = theVtx.z() - referencePos.z();
        double pz = totalP.z();
        double angleXYZ =
            (dx * px + dy * py + dz * pz) / (sqrt(dx * dx + dy * dy + dz * dz) * sqrt(px * px + py * py + pz * pz));
        if (angleXYZ < cosThetaXYZCut_)
          continue;
      }

      // calculate total energy of V0 3 ways: assume it's a kShort, a Lambda, or a LambdaBar.
      double piPlusE = sqrt(positiveP.mag2() + piMassSquared);
      double piMinusE = sqrt(negativeP.mag2() + piMassSquared);
      double protonE = sqrt(positiveP.mag2() + protonMassSquared);
      double antiProtonE = sqrt(negativeP.mag2() + protonMassSquared);
      double kShortETot = piPlusE + piMinusE;
      double lambdaEtot = protonE + piMinusE;
      double lambdaBarEtot = antiProtonE + piPlusE;

      // Create momentum 4-vectors for the 3 candidate types
      const reco::Particle::LorentzVector kShortP4(totalP.x(), totalP.y(), totalP.z(), kShortETot);
      const reco::Particle::LorentzVector lambdaP4(totalP.x(), totalP.y(), totalP.z(), lambdaEtot);
      const reco::Particle::LorentzVector lambdaBarP4(totalP.x(), totalP.y(), totalP.z(), lambdaBarEtot);

      reco::Particle::Point vtx(theVtx.x(), theVtx.y(), theVtx.z());
      const reco::Vertex::CovarianceMatrix vtxCov(theVtx.covariance());
      double vtxChi2(theVtx.chi2());
      double vtxNdof(theVtx.ndof());

      // Create the VertexCompositeCandidate object that will be stored in the Event
      reco::VertexCompositeCandidate* theKshort = nullptr;
      reco::VertexCompositeCandidate* theLambda = nullptr;
      reco::VertexCompositeCandidate* theLambdaBar = nullptr;

      if (doKShorts_) {
        theKshort = new reco::VertexCompositeCandidate(0, kShortP4, vtx, vtxCov, vtxChi2, vtxNdof);
      }
      if (doLambdas_) {
        if (positiveP.mag2() > negativeP.mag2()) {
          theLambda = new reco::VertexCompositeCandidate(0, lambdaP4, vtx, vtxCov, vtxChi2, vtxNdof);
        } else {
          theLambdaBar = new reco::VertexCompositeCandidate(0, lambdaBarP4, vtx, vtxCov, vtxChi2, vtxNdof);
        }
      }

      // Create daughter candidates for the VertexCompositeCandidates
      reco::RecoChargedCandidate thePiPlusCand(
          1, reco::Particle::LorentzVector(positiveP.x(), positiveP.y(), positiveP.z(), piPlusE), vtx);
      thePiPlusCand.setTrack(positiveTrackRef);

      reco::RecoChargedCandidate thePiMinusCand(
          -1, reco::Particle::LorentzVector(negativeP.x(), negativeP.y(), negativeP.z(), piMinusE), vtx);
      thePiMinusCand.setTrack(negativeTrackRef);

      reco::RecoChargedCandidate theProtonCand(
          1, reco::Particle::LorentzVector(positiveP.x(), positiveP.y(), positiveP.z(), protonE), vtx);
      theProtonCand.setTrack(positiveTrackRef);

      reco::RecoChargedCandidate theAntiProtonCand(
          -1, reco::Particle::LorentzVector(negativeP.x(), negativeP.y(), negativeP.z(), antiProtonE), vtx);
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
        addp4.set(*theLambda);
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
