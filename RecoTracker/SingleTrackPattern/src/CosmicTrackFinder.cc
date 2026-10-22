// Package:    RecoTracker/SingleTrackPattern
// Class:      CosmicTrackFinder
// Original Author:  Michele Pioppi-INFN perugia
#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/SingleTrackPattern/interface/CosmicTrackFinder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

namespace {
  /// convenience typedef for shorter syntax below
  template <typename collection>
  using optionalInput = std::optional<std::reference_wrapper<const collection>>;
}  // namespace

namespace cms {

  CosmicTrackFinder::CosmicTrackFinder(edm::ParameterSet const& conf)
      : cosmicTrajectoryBuilder_(conf, consumesCollector()), crackTrajectoryBuilder_(conf, consumesCollector()) {
    geometry = conf.getUntrackedParameter<std::string>("GeometricStructure", "STANDARD");
    useHitsSplitting_ = conf.getParameter<bool>("useHitsSplitting");
    matchedrecHitsToken_ =
        mayConsume<SiStripMatchedRecHit2DCollection>(conf.getParameter<edm::InputTag>("matchedRecHits"));
    rphirecHitsToken_ = mayConsume<SiStripRecHit2DCollection>(conf.getParameter<edm::InputTag>("rphirecHits"));
    stereorecHitsToken_ = mayConsume<SiStripRecHit2DCollection>(conf.getParameter<edm::InputTag>("stereorecHits"));
    pixelRecHitsToken_ = mayConsume<SiPixelRecHitCollection>(conf.getParameter<edm::InputTag>("pixelRecHits"));
    otRecHitsToken_ = mayConsume<Phase2TrackerRecHit1DCollectionNew>(conf.getParameter<edm::InputTag>("OTRecHits"));

    seedToken_ = consumes<TrajectorySeedCollection>(conf.getParameter<edm::InputTag>("cosmicSeeds"));

    produces<TrackCandidateCollection>();
  }

  // Virtual destructor needed.
  CosmicTrackFinder::~CosmicTrackFinder() {}

  // Functions that gets called by framework every event
  void CosmicTrackFinder::produce(edm::Event& e, const edm::EventSetup& es) {
    using namespace std;

    // retrieve seeds
    edm::Handle<TrajectorySeedCollection> seed;
    e.getByToken(seedToken_, seed);

    //retrieve PixelRecHits
    static const SiPixelRecHitCollection s_empty;
    const SiPixelRecHitCollection* pixelHitCollection = &s_empty;
    edm::Handle<SiPixelRecHitCollection> pixelHits;
    bool have_pixel = false;
    if (geometry != "MTCC" && (geometry != "CRACK")) {
      if (e.getByToken(pixelRecHitsToken_, pixelHits)) {
        pixelHitCollection = pixelHits.product();
        have_pixel = true;
      } else {
        Labels l;
        labelsForToken(pixelRecHitsToken_, l);
        edm::LogWarning("CosmicTrackFinder") << "Collection SiPixelRecHitCollection with InputTag " << l.module
                                             << " cannot be found, using empty collection of same type.";
      }
    }

    // retrieve StripRecHits
    edm::Handle<SiStripMatchedRecHit2DCollection> matchedrecHits;
    bool have_matchedrecHits = e.getByToken(matchedrecHitsToken_, matchedrecHits);
    edm::Handle<SiStripRecHit2DCollection> rphirecHits;
    bool have_rphirecHits = e.getByToken(rphirecHitsToken_, rphirecHits);
    edm::Handle<SiStripRecHit2DCollection> stereorecHits;
    bool have_stereorecHits = e.getByToken(stereorecHitsToken_, stereorecHits);
    edm::Handle<Phase2TrackerRecHit1DCollectionNew> otHitCollection;
    bool have_otRecHits = e.getByToken(otRecHitsToken_, otHitCollection);

    // check that the strip hits we find match the expectation
    // for one of the CMS running scenarios
    if (!(have_stereorecHits && have_matchedrecHits && have_rphirecHits) && !have_otRecHits) {
      Labels l1, l2, l3, l4;
      labelsForToken(matchedrecHitsToken_, l1);
      labelsForToken(rphirecHitsToken_, l2);
      labelsForToken(stereorecHitsToken_, l3);
      labelsForToken(otRecHitsToken_, l4);
      edm::LogWarning("CosmicTrackFinder")
          << "No valid strip hit collection found, tried the input tags" << std::endl
          << "   " << l1.module << ", " << l2.module << ", " << l3.module << " and " << l4.module << std::endl
          << " - either the first three (Phase-1) or the last (Phase-2) should exist.";
    }
    // Step B: create empty output collection
    auto output = std::make_unique<TrackCandidateCollection>();

    edm::LogVerbatim("CosmicTrackFinder") << "========== Cosmic Track Finder Info ==========";
    edm::LogVerbatim("CosmicTrackFinder") << " Numbers of Seeds " << (*seed).size();
    if (!(*seed).empty()) {
      std::vector<Trajectory> trajoutput;

      const TransientTrackingRecHitBuilder* hitBuilder = nullptr;
      if (geometry != "CRACK") {
        // to support both the Phase-1 and the Phase-2 layout,
        // check which collections are available and only
        // pass along those.
        optionalInput<SiStripRecHit2DCollection> stereo = std::nullopt;
        optionalInput<SiStripRecHit2DCollection> rphi = std::nullopt;
        optionalInput<SiStripMatchedRecHit2DCollection> matched = std::nullopt;
        optionalInput<SiPixelRecHitCollection> pix = std::nullopt;
        optionalInput<Phase2TrackerRecHit1DCollectionNew> ot = std::nullopt;

        if (have_pixel)
          pix = *pixelHitCollection;
        if (have_rphirecHits)
          rphi = *rphirecHits;
        if (have_stereorecHits)
          stereo = *stereorecHits;
        if (have_matchedrecHits)
          matched = *matchedrecHits;
        if (have_otRecHits)
          ot = *otHitCollection;

        cosmicTrajectoryBuilder_.run(*seed, stereo, rphi, matched, pix, ot, es, e, trajoutput);
        hitBuilder = cosmicTrajectoryBuilder_.hitBuilder();
      } else {
        /// CRack: So far, only support Phase-1 style hits
        crackTrajectoryBuilder_.run(
            *seed, *stereorecHits, *rphirecHits, *matchedrecHits, *pixelHitCollection, es, e, trajoutput);
        hitBuilder = crackTrajectoryBuilder_.hitBuilder();
      }
      assert(hitBuilder);
      Traj2TrackHits t2t(hitBuilder, true);

      edm::LogVerbatim("CosmicTrackFinder") << " Numbers of Temp Trajectories " << trajoutput.size();
      edm::LogVerbatim("CosmicTrackFinder") << "========== END Info ==========";
      if (!trajoutput.empty()) {
        // crazyness...
        std::vector<Trajectory*> tmpTraj;
        std::vector<Trajectory>::iterator itr;
        for (itr = trajoutput.begin(); itr != trajoutput.end(); itr++)
          tmpTraj.push_back(&(*itr));

        //The best track is selected
        //FOR MTCC the criteria are:
        //1)# of layers,2) # of Hits,3)Chi2
        if (geometry == "MTCC")
          stable_sort(tmpTraj.begin(), tmpTraj.end(), CompareTrajLay());
        else
          stable_sort(tmpTraj.begin(), tmpTraj.end(), CompareTrajChi());

        ///.....

        const Trajectory theTraj = *(*tmpTraj.begin());
        bool seedplus = (theTraj.seed().direction() == alongMomentum);

        // std::cout << "CosmicTrackFinder " <<"Reconstruction " <<  (seedplus ? "along" : "opposite to") << " momentum" << std::endl;
        LogDebug("CosmicTrackFinder") << "Reconstruction " << (seedplus ? "along" : "opposite to") << " momentum";

        /*
	// === the convention is to save always final tracks with hits sorted *along* momentum
        // --- this is NOT what was necessaraly happening before and not consistent with what done in standard CKF (this is a candidate not a track) 	
	*/
        edm::OwnVector<TrackingRecHit> recHits;
        if (theTraj.direction() == alongMomentum)
          std::cout << "cosmic: along momentum... " << std::endl;
        t2t(theTraj, recHits, useHitsSplitting_);
        recHits.reverse();  // according to original code

        /*
	Trajectory::RecHitContainer thits;
	//it->recHitsV(thits);
        theTraj.recHitsV(thits,useHitsSplitting_);
	edm::OwnVector<TrackingRecHit> recHits;
	recHits.reserve(thits.size());
	// reverse hit order
	for (Trajectory::RecHitContainer::const_iterator hitIt = thits.end()-1;
	     hitIt >= thits.begin(); hitIt--) {
	  recHits.push_back( (**hitIt).hit()->clone());
	}
       */

        TSOS firstState;
        unsigned int firstId;

        // assume not along momentum....
        firstState = theTraj.lastMeasurement().updatedState();
        firstId = theTraj.lastMeasurement().recHitR().rawId();
        //firstId = recHits.front().rawId();

        /*
	cout << "firstState y, z: " << firstState.globalPosition().y() 
	     << " , " << firstState.globalPosition().z() <<  endl;

	*/

        AlgebraicSymMatrix55 C = AlgebraicMatrixID();
        TSOS startingState(
            firstState.localParameters(), LocalTrajectoryError(C), firstState.surface(), firstState.magneticField());

        // protection againt invalid initial states
        if (!firstState.isValid()) {
          edm::LogWarning("CosmicTrackFinder") << "invalid innerState, will not make TrackCandidate";
          e.put(std::move(output));
          return;
        }

        // FIXME in case of slitting this can happen see CkfTrackCandidateMakerBase
        if (firstId != recHits.front().rawId()) {
          edm::LogWarning("CosmicTrackFinder") << "Mismatch in DetID of first hit: firstID= " << firstId
                                               << "   DetId= " << recHits.front().geographicalId().rawId();
          e.put(std::move(output));
          return;
        }

        PTrajectoryStateOnDet const& state = trajectoryStateTransform::persistentState(startingState, firstId);

        output->push_back(TrackCandidate(recHits, theTraj.seed(), state, theTraj.seedRef()));
      }
    }
    e.put(std::move(output));
  }

}  // namespace cms
