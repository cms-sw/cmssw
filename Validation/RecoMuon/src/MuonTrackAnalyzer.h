#ifndef Validation_RecoMuon_MuonTrackAnalyzer_H
#define Validation_RecoMuon_MuonTrackAnalyzer_H

/** \class MuonTrackAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  $Date: 2007/03/13 09:39:37 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco {class TransientTrack;}

class TFile;
class TH1F; 
class TH2F;
class HTrackVariables;
class HTrack;

class TrajectoryStateOnSurface;
class FreeTrajectoryState;
class MuonServiceProxy;
class MuonPatternRecoDumper;
class TrajectorySeed;
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
  void tracksAnalysis(const edm::Event & event, const edm::EventSetup& eventSetup,
		      edm::Handle<edm::SimTrackContainer> simTracks);
  void seedsAnalysis(const edm::Event & event, const edm::EventSetup& eventSetup,
		     edm::Handle<edm::SimTrackContainer> simTracks);
    

  virtual void beginJob(const edm::EventSetup& eventSetup) ;
  virtual void endJob() ;
 protected:

 private:
  bool isInTheAcceptance(double eta);
  bool checkMuonSimHitPresence(const edm::Event & event,
			       edm::Handle<edm::SimTrackContainer> simTracks);

  std::pair<SimTrack,double> getSimTrack(TrajectoryStateOnSurface &tsos,
				       edm::Handle<edm::SimTrackContainer> simTracks);  

  void fillPlots(const edm::Event &event, edm::Handle<edm::SimTrackContainer> &simTracks);
  void fillPlots(reco::TransientTrack &track, SimTrack &simTrack);
  void fillPlots(TrajectoryStateOnSurface &recoTSOS,SimTrack &simState,
		 HTrack*, MuonPatternRecoDumper&);
  void fillPlots(FreeTrajectoryState &recoFTS,SimTrack &simTrack,
		 HTrack *histo, MuonPatternRecoDumper &debug);


  TrajectoryStateOnSurface getSeedTSOS(const TrajectorySeed& seed);

  std::string theRootFileName;
  TFile* theFile;

  EtaRange theEtaRange;
  
  edm::InputTag theTracksLabel;
  edm::InputTag theSeedsLabel;
  edm::InputTag theCSCSimHitLabel;
  edm::InputTag theDTSimHitLabel; 
  edm::InputTag theRPCSimHitLabel;

  bool doTracksAnalysis;
  bool doSeedsAnalysis;
  std::string theSeedPropagatorName;

  MuonServiceProxy *theService;
  MuonUpdatorAtVertex *theUpdator;

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

  HTrack *hRecoSeedInner;
  HTrack *hRecoSeedPCA;
  HTrack *hRecoTracksPCA; 
  HTrack *hRecoTracksInner;
  HTrack *hRecoTracksOuter;

  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;
};
#endif

 
