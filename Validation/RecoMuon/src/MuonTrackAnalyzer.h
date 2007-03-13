#ifndef Validation_RecoMuon_MuonTrackAnalyzer_H
#define Validation_RecoMuon_MuonTrackAnalyzer_H

/** \class MuonTrackAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2006/09/01 14:35:48 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F; 
class TH2F;
class HTrackVariables;
class HTrack;

namespace reco {class TransientTrack;}

class TrajectoryStateOnSurface;
class MuonServiceProxy;
class MuonUpdatorAtVertex;

class MuonTrackAnalyzer: public edm::EDAnalyzer {

 public:
  enum EtaRange{all,barrel,endcap};

 public:
  /// Constructor
  MuonTrackAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~MuonTrackAnalyzer();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void beginJob(const edm::EventSetup& eventSetup) ;
  virtual void endJob() ;
 protected:

 private:
  bool isInTheAcceptance(double eta);
  bool checkMuonSimHitPresence(const edm::Event&);

  SimTrack getSimTrack(TrajectoryStateOnSurface &tsos,
		       edm::Handle<edm::SimTrackContainer> simTracks);

  SimTrack getSimTrack(FreeTrajectoryState &tsos,
		       edm::Handle<edm::SimTrackContainer> simTracks);
  

  std::string theRootFileName;
  TFile* theFile;

  edm::InputTag theDataType;
  EtaRange theEtaRange;
  
  edm::InputTag theMuonTrackLabel;
  edm::InputTag theSeedCollectionLabel;
  edm::InputTag theCSCSimHitLabel;
  edm::InputTag theDTSimHitLabel; 
  edm::InputTag theRPCSimHitLabel;

  MuonServiceProxy *theService;
  MuonUpdatorAtVertex *theUpdatorAtVtx;

  // Histograms
  TH1F *hChi2;
  TH1F *hChi2Norm;
  TH1F *hHitsPerTrack;
  TH1F *hDof;
  TH1F *hChi2Prob;

  TH1F *hNumberOfTracks;
  TH2F *hNumberOfTracksVsEta;
  TH2F *hChargeVsEta;
  TH2F *hChargeVsPt;
  TH2F *hPtRecVsPtGen;

  TH2F *hChi2VsEta;
  TH2F *hChi2NormVsEta;
  TH2F *hHitsPerTrackVsEta;
  TH2F *hDofVsEta; 
  TH2F *hChi2ProbVsEta;
  TH2F *hDeltaPtVsEta;
  TH2F *hDeltaPt_In_Out_VsEta;

  HTrackVariables *hSimTracks;

  HTrack *hRecoTracksExtrAtPCA;
  HTrack *hRecoTracksUpdatedAtVTX;

  HTrack *hRecoTracksVTX; 
  HTrack *hRecoTracksInner;
  HTrack *hRecoTracksOuter;
  
  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;
};
#endif

 
