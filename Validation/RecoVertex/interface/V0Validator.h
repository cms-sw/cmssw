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
// $Id: V0Validator.h,v 1.6 2010/02/25 20:15:42 drell Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/Records/interface/VertexAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
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
#include "RecoVertex/V0Producer/interface/V0Producer.h"

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

class V0Validator : public edm::EDAnalyzer {

public:
  explicit V0Validator(const edm::ParameterSet&);
  ~V0Validator();


private:
  //virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  //Quantities that are to be histogrammed
  float K0sGenEta, LamGenEta, K0sGenpT, LamGenpT, K0sGenR, LamGenR;
  float LamGenX, LamGenY, LamGenZ, KsGenX, KsGenY, KsGenZ;
  float K0sCandEta, LamCandEta, K0sCandpT, LamCandpT, K0sCandR, LamCandR;
  unsigned int K0sGenStatus, LamGenStatus, K0sCandStatus, LamCandStatus;
  unsigned int K0sPiCandStatus[2], LamPiCandStatus[2], K0sPiEff[2], LamPiEff[2];

  //Bookkeeping quantities
  int genLam, genK0s, realLamFound, realK0sFound, realLamFoundEff, realK0sFoundEff;
  int lamTracksFound, k0sTracksFound, lamCandFound, k0sCandFound, noTPforK0sCand, noTPforLamCand;

  //Temporary histograms so that we can divide them for efficiencies.
  //  They are turned into MonitorElements in endJob()
  /*  TH1F* ksEffVsRHist;
  TH1F* ksEffVsEtaHist;
  TH1F* ksEffVsPtHist;
  TH1F* ksTkEffVsRHist;
  TH1F* ksTkEffVsEtaHist;
  TH1F* ksTkEffVsPtHist;
  TH1F* ksFakeVsRHist;
  TH1F* ksFakeVsEtaHist;
  TH1F* ksFakeVsPtHist;
  TH1F* ksTkFakeVsRHist;
  TH1F* ksTkFakeVsEtaHist;
  TH1F* ksTkFakeVsPtHist;

  TH1F* ksEffVsRHist_denom;
  TH1F* ksEffVsEtaHist_denom;
  TH1F* ksEffVsPtHist_denom;
  TH1F* ksFakeVsRHist_denom;
  TH1F* ksFakeVsEtaHist_denom;
  TH1F* ksFakeVsPtHist_denom;

  TH1F* ksXResolutionHist;
  TH1F* ksYResolutionHist;
  TH1F* ksZResolutionHist;
  TH1F* ksAbsoluteDistResolutionHist;
  TH1F* lamXResolutionHist;
  TH1F* lamYResolutionHist;
  TH1F* lamZResolutionHist;
  TH1F* lamAbsoluteDistResolutionHist;

  TH1F* lamEffVsRHist;
  TH1F* lamEffVsEtaHist;
  TH1F* lamEffVsPtHist;
  TH1F* lamTkEffVsRHist;
  TH1F* lamTkEffVsEtaHist;
  TH1F* lamTkEffVsPtHist;
  TH1F* lamFakeVsRHist;
  TH1F* lamFakeVsEtaHist;
  TH1F* lamFakeVsPtHist;
  TH1F* lamTkFakeVsRHist;
  TH1F* lamTkFakeVsEtaHist;
  TH1F* lamTkFakeVsPtHist;

  TH1F* lamEffVsRHist_denom;
  TH1F* lamEffVsEtaHist_denom;
  TH1F* lamEffVsPtHist_denom;
  TH1F* lamFakeVsRHist_denom;
  TH1F* lamFakeVsEtaHist_denom;
  TH1F* lamFakeVsPtHist_denom;

  TH1F* nKsHist;
  TH1F* nLamHist;

  TH1F* lamCandStatusHist;
  TH1F* ksCandStatusHist;

  TH1F* fakeKsMassHisto;
  TH1F* goodKsMassHisto;
  TH1F* fakeLamMassHisto;
  TH1F* goodLamMassHisto;

  TH1F* ksFakeDauRadDistHisto;
  TH1F* lamFakeDauRadDistHisto;*/

  // DQMStore and MonitorElements for final histograms
  DQMStore* theDQMstore;

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
  edm::InputTag k0sCollectionTag;
  edm::InputTag lamCollectionTag;
  std::string dirName;

};

