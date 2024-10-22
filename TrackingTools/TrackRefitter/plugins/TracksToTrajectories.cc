#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformerForGlobalCosmicMuons.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformerForCosmicMuons.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"

/** \class TracksToTrajectories
 *  This class, which is a EDProducer, takes a reco::TrackCollection from the Event and refits the rechits 
 *  strored in the reco::Tracks. The final result is a std::vector of Trajectories (objs of the type "Trajectory"), 
 *  which is loaded into the Event in a transient way
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// In principle this should be anonymous namespace, but then
// DEFINE_FWK_MODULE() will yield compilation warnings, so using
// (hopefully) unique namespace instead.
// The point of the namespace is to not to pollute the global
// namespace (and symbol space).
namespace tracksToTrajectories {
  struct Count {
    Count() : theNTracks(0), theNFailures(0) {}
    //Using mutable since we want to update the value.
    mutable std::atomic<int> theNTracks;
    mutable std::atomic<int> theNFailures;
  };
}  // namespace tracksToTrajectories
using namespace tracksToTrajectories;

class TracksToTrajectories : public edm::stream::EDProducer<edm::GlobalCache<Count>> {
public:
  /// Constructor
  TracksToTrajectories(const edm::ParameterSet&, const Count*);

  /// Destructor
  ~TracksToTrajectories() override;

  static std::unique_ptr<Count> initializeGlobalCache(edm::ParameterSet const&) { return std::make_unique<Count>(); }

  // Operations
  static void globalEndJob(Count const* iCount);

  /// Convert a reco::TrackCollection into std::vector<Trajectory>
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<reco::TrackCollection> theTracksToken;
  std::unique_ptr<TrackTransformerBase> theTrackTransformer;
};

using namespace std;
using namespace edm;

/// Constructor
TracksToTrajectories::TracksToTrajectories(const ParameterSet& parameterSet, const Count*) {
  theTracksToken = consumes<reco::TrackCollection>(parameterSet.getParameter<InputTag>("Tracks"));

  ParameterSet trackTransformerParam = parameterSet.getParameter<ParameterSet>("TrackTransformer");

  string type = parameterSet.getParameter<string>("Type");

  if (type == "Default")
    theTrackTransformer = std::make_unique<TrackTransformer>(trackTransformerParam, consumesCollector());
  else if (type == "GlobalCosmicMuonsForAlignment")
    theTrackTransformer =
        std::make_unique<TrackTransformerForGlobalCosmicMuons>(trackTransformerParam, consumesCollector());
  else if (type == "CosmicMuonsForAlignment")
    theTrackTransformer = std::make_unique<TrackTransformerForCosmicMuons>(trackTransformerParam, consumesCollector());
  else {
    throw cms::Exception("TracksToTrajectories")
        << "The selected algorithm does not exist"
        << "\n"
        << "Possible choices are:"
        << "\n"
        << "Type = [Default, GlobalCosmicMuonsForAlignment, CosmicMuonsForAlignment]";
  }

  produces<vector<Trajectory>>("Refitted");
  produces<TrajTrackAssociationCollection>("Refitted");
}

/// Destructor
TracksToTrajectories::~TracksToTrajectories() {}

void TracksToTrajectories::globalEndJob(Count const* iCount) {
  constexpr char metname[] = "Reco|TrackingTools|TracksToTrajectories";

  auto theNFailures = iCount->theNFailures.load();
  auto theNTracks = iCount->theNTracks.load();

  if (theNFailures != 0)
    LogWarning(metname) << "During the refit there were " << theNFailures << " out of " << theNTracks
                        << " tracks, i.e. failure rate is: " << double(theNFailures) / theNTracks;
  else {
    LogTrace(metname) << "Refit of the tracks done without any failure";
  }
}

/// Convert Tracks into Trajectories
void TracksToTrajectories::produce(Event& event, const EventSetup& setup) {
#ifdef EDM_ML_DEBUG
  constexpr char metname[] = "Reco|TrackingTools|TracksToTrajectories";
#endif

  theTrackTransformer->setServices(setup);

  // Collection of Trajectory
  auto trajectoryCollection = std::make_unique<vector<Trajectory>>();

  // Get the reference
  RefProd<vector<Trajectory>> trajectoryCollectionRefProd = event.getRefBeforePut<vector<Trajectory>>("Refitted");

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> tracks;
  event.getByToken(theTracksToken, tracks);

  // Association map between Trajectory and Track
  auto trajTrackMap = std::make_unique<TrajTrackAssociationCollection>(trajectoryCollectionRefProd, tracks);

  Ref<vector<Trajectory>>::key_type trajectoryIndex = 0;
  reco::TrackRef::key_type trackIndex = 0;

  // Loop over the Rec tracks
  for (auto const& newTrack : *tracks) {
    ++(globalCache()->theNTracks);

    auto const& trajectoriesSM = theTrackTransformer->transform(newTrack);

    if (!trajectoriesSM.empty()) {
      // Load the trajectory in the Trajectory Container
      trajectoryCollection->push_back(trajectoriesSM.front());

      // Make the association between the Trajectory and the original Track
      trajTrackMap->insert(Ref<vector<Trajectory>>(trajectoryCollectionRefProd, trajectoryIndex++),
                           reco::TrackRef(tracks, trackIndex++));
    } else {
      LogTrace(metname) << "Error in the Track refitting. This should not happen";
      ++(globalCache()->theNFailures);
    }
  }
  LogTrace(metname) << "Load the Trajectory Collection";
  event.put(std::move(trajectoryCollection), "Refitted");
  event.put(std::move(trajTrackMap), "Refitted");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TracksToTrajectories);
