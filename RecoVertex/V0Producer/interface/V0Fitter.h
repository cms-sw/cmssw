// -*- C++ -*-
//
// Package:    V0Producer
// Class:      V0Fitter
// 
/**\class V0Fitter V0Fitter.h RecoVertex/V0Producer/interface/V0Fitter.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Fri May 18 22:57:40 CEST 2007
// $Id: V0Fitter.h,v 1.2 2007/07/09 12:58:51 drell Exp $
//
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <string>


class V0Fitter {
 public:
  V0Fitter(const edm::Event& iEvent, const edm::EventSetup& iSetup,
	   std::string trackRecoAlgo, const int useRefittedTrax,
	   const int storeRefittedTrax, const double chi2Cut, 
	   const double rVtxCut, const double vtxSigCut, 
	   const double collinCut, const double kShortMassCut,
	   const double lambdaMassCut, 
	   const int doKshort, const int doLambda);
  ~V0Fitter();

  // Get methods for the VertexCollections
  //reco::VertexCollection getKshortCollection() const;
  //reco::VertexCollection getLambdaCollection() const;
  //reco::VertexCollection getLambdaBarCollection() const;

  // Switching to L. Lista's reco::Candidate infrastructure for V0 storage
  std::vector<reco::V0Candidate> getKshorts() const;
  std::vector<reco::V0Candidate> getLambdas() const;
  std::vector<reco::V0Candidate> getLambdaBars() const;

 private:
  // VertexCollection objects that get filled by the fitAll method
  //reco::VertexCollection K0s;
  //reco::VertexCollection Lam0;
  //reco::VertexCollection Lam0Bar;

  // STL vector of V0Candidate that will be filled with V0Candidates by fitAll()
  std::vector<reco::V0Candidate> theKshorts;
  std::vector<reco::V0Candidate> theLambdas;
  std::vector<reco::V0Candidate> theLambdaBars;

  // Vector used to temporarily hold candidates before cuts and selection
  std::vector<reco::V0Candidate> preCutCands;

  // Tracker geometry for discerning hit positions
  const TrackerGeometry* trackerGeom;

  std::string recoAlg;
  int useRefTrax;
  int storeRefTrax;
  int doKshorts;
  int doLambdas;

  // Cuts
  double chi2Cut;
  double rVtxCut;
  double vtxSigCut;
  double collinCut;
  double kShortMassCut;
  double lambdaMassCut;

  // Helper method that does the actual fitting using the KalmanVertexFitter
  void fitAll(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  // Applies cuts to the track collection before fitting vertices to
  //  each oppositely-charged track pair
  void applyPreFitCuts(std::vector<reco::Track> &tracks);

  // Applies cuts to theCands after vertices have been fit.
  //  This method should probably also contain the way to differentiate
  //  between Kshorts and lambdas, writing the PDGid to the Particle once
  //  this cut has been calculated.
  void applyPostFitCuts();
};
