/** \class TrackReader
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "TFile.h"
#include "TH1F.h"

#include <vector>

class TrackReader : public edm::one::EDAnalyzer<> {
public:
  typedef std::vector<Trajectory> Trajectories;

public:
  /// Constructor
  TrackReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~TrackReader();

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

protected:
  //  void printTrackRecHits(const reco::Track &, edm::ESHandle<GlobalTrackingGeometry>) const;

private:
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> trackerBuilderToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> muonBuilderToken_;
  edm::InputTag theInputLabel;
};

/// Constructor
TrackReader::TrackReader(const edm::ParameterSet& parameterSet)
    : trackerBuilderToken_(
          esConsumes(edm::ESInputTag("", parameterSet.getParameter<std::string>("TrackerRecHitBuilder")))),
      muonBuilderToken_(esConsumes(edm::ESInputTag("", parameterSet.getParameter<std::string>("MuonRecHitBuilder")))) {
  theInputLabel = parameterSet.getParameter<edm::InputTag>("InputLabel");
}

/// Destructor
TrackReader::~TrackReader() = default;

void TrackReader::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  const TransientTrackingRecHitBuilder* theTrackerRecHitBuilder = &setup.getData(trackerBuilderToken_);
  const TransientTrackingRecHitBuilder* theMuonRecHitBuilder = &setup.getData(muonBuilderToken_);

  const std::string metname = "Reco|TrackingTools|TrackReader";

  // Get the RecTrack collection from the event
  edm::Handle<reco::TrackCollection> tracks;
  event.getByLabel(theInputLabel, tracks);

  for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
    for (trackingRecHit_iterator hit = track->recHitsBegin(); hit != track->recHitsEnd(); ++hit) {
      if ((*hit)->isValid()) {
        if ((*hit)->geographicalId().det() == DetId::Tracker) {
          LogDebug("TrackReader") << "Tracker hit";
          TransientTrackingRecHit::RecHitPointer tthit = theTrackerRecHitBuilder->build(&**hit);
          //	  TransientTrackingRecHit::RecHitPointer preciseHit = tthit.clone(predTsos);
          LogTrace("TrackReader") << "Position: " << tthit->globalPosition();
        } else if ((*hit)->geographicalId().det() == DetId::Muon) {
          LogDebug("TrackReader") << "Muon hit";
          TransientTrackingRecHit::RecHitPointer tthit = theMuonRecHitBuilder->build(&**hit);
          LogTrace("TrackReader") << "Position: " << tthit->globalPosition();
        }
      }
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackReader);
