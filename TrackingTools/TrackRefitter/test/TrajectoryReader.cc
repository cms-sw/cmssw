/** \class TrajectoryReader
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */
// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "TFile.h"
#include "TH1F.h"

#include <vector>

class TrajectoryReader : public edm::one::EDAnalyzer<> {
public:
  typedef std::vector<Trajectory> Trajectories;

public:
  /// Constructor
  TrajectoryReader(const edm::ParameterSet &pset);

  /// Destructor
  virtual ~TrajectoryReader();

  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup);

  // Operations
  void beginJob();
  void endJob();

protected:
  void printTrajectoryRecHits(const Trajectory &, edm::ESHandle<GlobalTrackingGeometry>) const;
  void printTrackRecHits(const reco::Track &, edm::ESHandle<GlobalTrackingGeometry>) const;

private:
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theGlobGeomToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMFToken;

  edm::EDGetTokenT<Trajectories> theTrajToken;
  edm::EDGetTokenT<reco::TrackCollection> theTrackToken;
  edm::EDGetTokenT<TrajTrackAssociationCollection> theAssocToken;

  edm::InputTag theInputLabel;
  TFile *theFile;
  std::string theRootFileName;

  TH1F *hDPtIn;
  TH1F *hDPtOut;
  TH1F *hSuccess;
  TH1F *hNHitLost;
  TH1F *hFractionHitLost;
};

/// Constructor
TrajectoryReader::TrajectoryReader(const edm::ParameterSet &pset)
    : theGlobGeomToken(esConsumes()), theMFToken(esConsumes()) {
  theInputLabel = pset.getParameter<edm::InputTag>("InputLabel");

  theTrajToken = consumes<Trajectories>(theInputLabel), theTrackToken = consumes<reco::TrackCollection>(theInputLabel),
  theAssocToken = consumes<TrajTrackAssociationCollection>(theInputLabel);

  theRootFileName = pset.getUntrackedParameter<std::string>("rootFileName");
}

/// Destructor
TrajectoryReader::~TrajectoryReader() = default;

// Operations
void TrajectoryReader::beginJob() {
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  hDPtIn = new TH1F("DeltaPtIn", "P_{t}^{Track}-P_{t}^{Traj} inner state", 10000, -20, 20);
  hDPtOut = new TH1F("DeltaPtOut", "P_{t}^{Track}-P_{t}^{Traj} outer state", 10000, -20, 20);

  hNHitLost = new TH1F("NHitLost", "Number of lost hits", 100, 0, 100);
  hFractionHitLost = new TH1F("FractionHitLost", "Fraction of lost hits", 100, 0, 100);
  hSuccess = new TH1F("Success", "Number of Success", 2, 0, 2);
}

void TrajectoryReader::endJob() {
  theFile->cd();

  // Write the histos to file
  hDPtIn->Write();
  hDPtOut->Write();
  hNHitLost->Write();
  hFractionHitLost->Write();
  hSuccess->Write();
  theFile->Close();
}

void TrajectoryReader::printTrajectoryRecHits(const Trajectory &trajectory,
                                              edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const {
  const std::string metname = "Reco|TrackingTools|TrajectoryReader";

  TransientTrackingRecHit::ConstRecHitContainer rechits = trajectory.recHits();
  LogDebug(metname) << "Size of the RecHit container: " << rechits.size();

  int i = 0;
  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator recHit = rechits.begin(); recHit != rechits.end();
       ++recHit)
    if ((*recHit)->isValid()) {
      const GeomDet *geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
      double r = geomDet->surface().position().perp();
      double z = geomDet->toGlobal((*recHit)->localPosition()).z();
      LogTrace(metname) << i++ << " r: " << r << " z: " << z << " " << geomDet->toGlobal((*recHit)->localPosition())
                        << std::endl;
    }
}

void TrajectoryReader::printTrackRecHits(const reco::Track &track,
                                         edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const {
  const std::string metname = "Reco|TrackingTools|TrajectoryReader";

  LogTrace(metname) << "Valid RecHits: " << track.found() << " invalid RecHits: " << track.lost();

  int i = 0;
  for (trackingRecHit_iterator recHit = track.recHitsBegin(); recHit != track.recHitsEnd(); ++recHit)
    if ((*recHit)->isValid()) {
      const GeomDet *geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
      double r = geomDet->surface().position().perp();
      double z = geomDet->surface().position().z();
      LogTrace(metname) << i++ << " GeomDet position r: " << r << " z: " << z;
    }
}

void TrajectoryReader::analyze(const edm::Event &event, const edm::EventSetup &eventSetup) {
  // Global Tracking Geometry
  const GlobalTrackingGeometry *trackingGeometry = &eventSetup.getData(theGlobGeomToken);

  // Magfield Field
  const MagneticField *magField = &eventSetup.getData(theMFToken);

  const std::string metname = "Reco|TrackingTools|TrajectoryReader";

  // Get the RecTrack collection from the event
  const reco::TrackCollection tracks = event.get(theTrackToken);

  if (tracks.empty())
    return;

  // Get the Trajectory collection from the event
  const Trajectories trajectories = event.get(theTrajToken);

  LogTrace(metname) << "looking at: " << theInputLabel;

  LogTrace(metname) << "All trajectories";
  for (const auto &trajectory : trajectories) {
    printTrajectoryRecHits(trajectory, trackingGeometry);
  }

  LogTrace(metname) << "All tracks";
  for (const auto &tr : tracks) {
    printTrackRecHits(tr, trackingGeometry);
  }

  const TrajTrackAssociationCollection assoMap = event.get(theAssocToken);

  LogTrace(metname) << "Association";
  //for (TrajTrackAssociationCollection::const_iterator it = assoMap.begin(); it != assoMap.end(); ++it) {
  for (const auto &it : assoMap) {
    const edm::Ref<std::vector<Trajectory> > traj = it.key;
    const reco::TrackRef tk = it.val;

    printTrackRecHits(*tk, trackingGeometry);
    printTrajectoryRecHits(*traj, trackingGeometry);

    // Check the difference in Pt
    reco::TransientTrack track(tk, &*magField, trackingGeometry);

    hDPtIn->Fill(track.innermostMeasurementState().globalMomentum().perp() -
                 traj->lastMeasurement().updatedState().globalMomentum().perp());
    hDPtOut->Fill(track.outermostMeasurementState().globalMomentum().perp() -
                  traj->firstMeasurement().updatedState().globalMomentum().perp());

    int diff = track.recHitsSize() - traj->recHits().size();
    LogTrace(metname) << "Difference: " << diff;
    hNHitLost->Fill(diff);
    hFractionHitLost->Fill(double(diff) / track.recHitsSize());
  }

  int traj_size = trajectories.size();
  int track_size = tracks.size();

  if (traj_size != track_size) {
    LogTrace(metname) << "Mismatch between the # of Tracks (" << track_size << ") and the # of Trajectories! ("
                      << traj_size << ") in " << event.id();
    hSuccess->Fill(0);
  } else
    hSuccess->Fill(1);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrajectoryReader);
