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
// $Id: V0Fitter.h,v 1.25 2013/04/11 23:17:44 wmtan Exp $
//
//

#ifndef RECOVERTEX__V0_FITTER_H
#define RECOVERTEX__V0_FITTER_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include <string>
#include <fstream>


class V0Fitter {
 public:
  V0Fitter(const edm::ParameterSet& theParams,
	   const edm::Event& iEvent, const edm::EventSetup& iSetup);
  ~V0Fitter();

  // Switching to L. Lista's reco::Candidate infrastructure for V0 storage
  const reco::VertexCompositeCandidateCollection& getKshorts() const;
  const reco::VertexCompositeCandidateCollection& getLambdas() const;

 private:
  // STL vector of VertexCompositeCandidate that will be filled with VertexCompositeCandidates by fitAll()
  reco::VertexCompositeCandidateCollection theKshorts;
  reco::VertexCompositeCandidateCollection theLambdas;

  // Tracker geometry for discerning hit positions
  const TrackerGeometry* trackerGeom;

  const MagneticField* magField;

  edm::InputTag recoAlg;
  bool useRefTrax;
  bool storeRefTrax;
  bool doKshorts;
  bool doLambdas;

  /*bool doPostFitCuts;
    bool doTkQualCuts;*/

  // Cuts
  double chi2Cut;
  double tkChi2Cut;
  int tkNhitsCut;
  double rVtxCut;
  double vtxSigCut;
  double vtxSigCut3D;
  double collinCut;
  double kShortMassCut;
  double lambdaMassCut;
  double impactParameterSigCut;
  double mPiPiCut;
  double tkDCACut;
  double innerHitPosCut;

  std::vector<reco::TrackBase::TrackQuality> qualities;

  edm::InputTag vtxFitter;

  // Helper method that does the actual fitting using the KalmanVertexFitter
  void fitAll(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  double findV0MassError(const GlobalPoint &vtxPos, const std::vector<reco::TransientTrack> &dauTracks);

  // Applies cuts to the VertexCompositeCandidates after they are fitted/created.
  //void applyPostFitCuts();

  // Stuff for debug file output.
  std::ofstream mPiPiMassOut;

  inline void initFileOutput() {
    mPiPiMassOut.open("mPiPi.txt", std::ios::app);
  }
  inline void cleanupFileOutput() {
    mPiPiMassOut.close();
  }
};

#endif
