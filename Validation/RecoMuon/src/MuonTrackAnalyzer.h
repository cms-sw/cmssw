#ifndef Validation_RecoMuon_MuonTrackAnalyzer_H
#define Validation_RecoMuon_MuonTrackAnalyzer_H

/** \class MuonTrackAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm
namespace reco {
  class TransientTrack;
}

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

class MuonTrackAnalyzer : public DQMEDAnalyzer {
public:
  enum EtaRange { all, barrel, endcap };

public:
  /// Constructor
  MuonTrackAnalyzer(const edm::ParameterSet &pset);

  /// Destructor
  ~MuonTrackAnalyzer() override;

  // Operations

  void beginJob() override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;
  void tracksAnalysis(const edm::Event &event,
                      const edm::EventSetup &eventSetup,
                      edm::Handle<edm::SimTrackContainer> simTracks);
  void seedsAnalysis(const edm::Event &event,
                     const edm::EventSetup &eventSetup,
                     edm::Handle<edm::SimTrackContainer> simTracks);

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

protected:
private:
  bool isInTheAcceptance(double eta);
  bool checkMuonSimHitPresence(const edm::Event &event, edm::Handle<edm::SimTrackContainer> simTracks);

  std::pair<SimTrack, double> getSimTrack(TrajectoryStateOnSurface &tsos,
                                          edm::Handle<edm::SimTrackContainer> simTracks);

  void fillPlots(const edm::Event &event, edm::Handle<edm::SimTrackContainer> &simTracks);
  void fillPlots(reco::TransientTrack &track, SimTrack &simTrack);
  void fillPlots(TrajectoryStateOnSurface &recoTSOS, SimTrack &simState, HTrack *, MuonPatternRecoDumper &);
  void fillPlots(FreeTrajectoryState &recoFTS, SimTrack &simTrack, HTrack *histo, MuonPatternRecoDumper &debug);

  TrajectoryStateOnSurface getSeedTSOS(const TrajectorySeed &seed);

  DQMStore *dbe_;
  std::string dirName_;

  std::string out;
  std::string subsystemname_;
  edm::ParameterSet pset;
  //TFile* theFile;

  EtaRange theEtaRange;

  edm::InputTag theSimTracksLabel;
  edm::InputTag theSeedsLabel;
  edm::InputTag theTracksLabel;
  edm::InputTag theCSCSimHitLabel;
  edm::InputTag theDTSimHitLabel;
  edm::InputTag theRPCSimHitLabel;

  edm::EDGetTokenT<edm::SimTrackContainer> theSimTracksToken;
  edm::EDGetTokenT<TrajectorySeedCollection> theSeedsToken;
  edm::EDGetTokenT<reco::TrackCollection> theTracksToken;
  edm::EDGetTokenT<std::vector<PSimHit> > theCSCSimHitToken;
  edm::EDGetTokenT<std::vector<PSimHit> > theDTSimHitToken;
  edm::EDGetTokenT<std::vector<PSimHit> > theRPCSimHitToken;

  bool doTracksAnalysis;
  bool doSeedsAnalysis;
  std::string theSeedPropagatorName;

  MuonServiceProxy *theService;
  MuonUpdatorAtVertex *theUpdator;

  // Histograms
  MonitorElement *hChi2;
  MonitorElement *hChi2Norm;
  MonitorElement *hHitsPerTrack;
  MonitorElement *hDof;
  MonitorElement *hChi2Prob;

  MonitorElement *hNumberOfTracks;
  MonitorElement *hNumberOfTracksVsEta;
  MonitorElement *hChargeVsEta;
  MonitorElement *hChargeVsPt;
  MonitorElement *hPtRecVsPtGen;

  MonitorElement *hChi2VsEta;
  MonitorElement *hChi2NormVsEta;
  MonitorElement *hHitsPerTrackVsEta;
  MonitorElement *hDofVsEta;
  MonitorElement *hChi2ProbVsEta;
  MonitorElement *hDeltaPtVsEta;
  MonitorElement *hDeltaPt_In_Out_VsEta;

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
