// -*- C++ -*-
//
// Package:    V0Validator
// Class:      V0Validator
// 
/**\class V0Validator V0Validator.cc Validation/RecoVertex/interface/V0Validator.h

 Description: Creates validation histograms for RecoVertex/V0Producer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Wed Feb 18 17:21:04 MST 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/VertexAssociatorRecord.h"
#include "SimTracker/VertexAssociation/interface/VertexAssociatorBase.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"

#include "TROOT.h"
#include "TMath.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"

class V0Validator : public DQMEDAnalyzer {

public:
  explicit V0Validator(const edm::ParameterSet&);
  ~V0Validator();


private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  //Quantities that are to be histogrammed
  float K0sGenEta, LamGenEta, K0sGenpT, LamGenpT, K0sGenR, LamGenR;
  float LamGenX, LamGenY, LamGenZ, KsGenX, KsGenY, KsGenZ;
  float K0sCandEta, LamCandEta, K0sCandpT, LamCandpT, K0sCandR, LamCandR;
  unsigned int K0sGenStatus, LamGenStatus, K0sCandStatus, LamCandStatus;
  unsigned int K0sPiCandStatus[2], LamPiCandStatus[2], K0sPiEff[2], LamPiEff[2];

  //Bookkeeping quantities
  int genLam, genK0s, realLamFound, realK0sFound, realLamFoundEff, realK0sFoundEff;
  int lamTracksFound, k0sTracksFound, lamCandFound, k0sCandFound, noTPforK0sCand, noTPforLamCand;

  // MonitorElements for final histograms
  MonitorElement* ksEffVsR;
  MonitorElement* ksEffVsEta;
  MonitorElement* ksEffVsPt;
  MonitorElement* ksTkEffVsR;
  MonitorElement* ksTkEffVsEta;
  MonitorElement* ksTkEffVsPt;
  MonitorElement* ksFakeVsR;
  MonitorElement* ksFakeVsEta;
  MonitorElement* ksFakeVsPt;
  MonitorElement* ksTkFakeVsR;
  MonitorElement* ksTkFakeVsEta;
  MonitorElement* ksTkFakeVsPt;

  MonitorElement* ksEffVsR_num;
  MonitorElement* ksEffVsEta_num;
  MonitorElement* ksEffVsPt_num;
  MonitorElement* ksTkEffVsR_num;
  MonitorElement* ksTkEffVsEta_num;
  MonitorElement* ksTkEffVsPt_num;
  MonitorElement* ksFakeVsR_num;
  MonitorElement* ksFakeVsEta_num;
  MonitorElement* ksFakeVsPt_num;
  MonitorElement* ksTkFakeVsR_num;
  MonitorElement* ksTkFakeVsEta_num;
  MonitorElement* ksTkFakeVsPt_num;

  MonitorElement* ksFakeVsR_denom;
  MonitorElement* ksFakeVsEta_denom;
  MonitorElement* ksFakeVsPt_denom;
  MonitorElement* ksEffVsR_denom;
  MonitorElement* ksEffVsEta_denom;
  MonitorElement* ksEffVsPt_denom;

  MonitorElement* lamFakeVsR_denom;
  MonitorElement* lamFakeVsEta_denom;
  MonitorElement* lamFakeVsPt_denom;
  MonitorElement* lamEffVsR_denom;
  MonitorElement* lamEffVsEta_denom;
  MonitorElement* lamEffVsPt_denom;

  MonitorElement* lamEffVsR;
  MonitorElement* lamEffVsEta;
  MonitorElement* lamEffVsPt;
  MonitorElement* lamTkEffVsR;
  MonitorElement* lamTkEffVsEta;
  MonitorElement* lamTkEffVsPt;
  MonitorElement* lamFakeVsR;
  MonitorElement* lamFakeVsEta;
  MonitorElement* lamFakeVsPt;
  MonitorElement* lamTkFakeVsR;
  MonitorElement* lamTkFakeVsEta;
  MonitorElement* lamTkFakeVsPt;

  MonitorElement* lamEffVsR_num;
  MonitorElement* lamEffVsEta_num;
  MonitorElement* lamEffVsPt_num;
  MonitorElement* lamTkEffVsR_num;
  MonitorElement* lamTkEffVsEta_num;
  MonitorElement* lamTkEffVsPt_num;
  MonitorElement* lamFakeVsR_num;
  MonitorElement* lamFakeVsEta_num;
  MonitorElement* lamFakeVsPt_num;
  MonitorElement* lamTkFakeVsR_num;
  MonitorElement* lamTkFakeVsEta_num;
  MonitorElement* lamTkFakeVsPt_num;

  MonitorElement* ksXResolution;
  MonitorElement* ksYResolution;
  MonitorElement* ksZResolution;
  MonitorElement* ksAbsoluteDistResolution;
  MonitorElement* lamXResolution;
  MonitorElement* lamYResolution;
  MonitorElement* lamZResolution;
  MonitorElement* lamAbsoluteDistResolution;

  MonitorElement* nKs;
  MonitorElement* nLam;

  MonitorElement* ksCandStatus;
  MonitorElement* lamCandStatus;

  MonitorElement* fakeKsMass;
  MonitorElement* goodKsMass;
  MonitorElement* fakeLamMass;
  MonitorElement* goodLamMass;

  MonitorElement* ksMassAll;
  MonitorElement* lamMassAll;


  MonitorElement* ksFakeDauRadDist;
  MonitorElement* lamFakeDauRadDist;


  std::string theDQMRootFileName;
  std::string dirName;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoRecoToSimCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> recoSimToRecoCollectionToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollection_Eff_Token_, trackingParticleCollectionToken_;
  edm::EDGetTokenT< edm::View<reco::Track> > edmView_recoTrack_Token_;
  edm::EDGetTokenT<edm::SimTrackContainer> edmSimTrackContainerToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> edmSimVertexContainerToken_;
  edm::EDGetTokenT< std::vector<reco::Vertex> > vec_recoVertex_Token_;
  edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> recoVertexCompositeCandidateCollection_k0s_Token_, recoVertexCompositeCandidateCollection_lambda_Token_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> recoTrackToTrackingParticleAssociator_Token_;
};

