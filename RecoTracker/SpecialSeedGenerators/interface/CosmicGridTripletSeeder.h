#pragma once

// system include files
#include <cmath>
#include <memory>
#include <vector>

// user include files
#include "RecoTracker/SpecialSeedGenerators/interface/CartesianSeedingGrid.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplet.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"


/// Grid triplet seeder for cosmic muons. 
/// Intended as a very simple and robust seeder for initial 
/// Phase-2 alignment studies in a non-collision environment
/// Forms triplets by connecting adjacent space-points in global y, 
/// using stubs in the strips to reduce combinatorics, and 
/// additionally binning the 3D space into cells and only 
/// using plausible cell combinations in the seeding.  
///
/// Also compatible with Run-3 tracker, without reconfiguration. 
class CosmicGridTripletSeeder : public edm::stream::EDProducer<> {
public:
    explicit CosmicGridTripletSeeder(const edm::ParameterSet& par);
    ~CosmicGridTripletSeeder() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::Event &e, const edm::EventSetup &c) override;

private:

    /// Encapsulated event state, for convenient passing between methods 
    /// and allowing for concurrent calls to the same producer instance 
    /// across multiple in-progress events. 
    struct TripletSeederEventState{
        CartesianSeedingGrid grid;  /// the seeding grid 
        std::map<const VectorHit*, std::vector<const BaseTrackerRecHit*>> vhConstituents;  /// the reco hits comprising each of the vector hits in the grid  
        const MagneticField *magfield = nullptr; /// cached magnetic field instance
        const TrackerGeometry *tracker = nullptr; /// cached tracking geometry  
        TkClonerImpl cloner;  /// cloner, used in the final seed fitting  
        std::unique_ptr<KFUpdator> theUpdator = nullptr; /// kalman updator 
        std::unique_ptr<PropagatorWithMaterial> thePropagatorAl = nullptr; /// propagator - along momentum 
        std::unique_ptr<PropagatorWithMaterial> thePropagatorOp = nullptr; /// propagator - opposite momentum 
    };

     /// @brief load conditions required for the event
     /// @param c: EventSetup instance 
     /// @return an event state instance with event-setup dependent members
     /// initialised based on the passed argument 
     TripletSeederEventState initEventState(const edm::EventSetup &c) const;

     /// @brief populate the seeding grid. 
     /// Will retrieve the hit collections and fill them into the seeding grid 
     /// contained within the event state.  
     /// @param e: Event to process
     /// @param state: State instance to update - will get its grid and VH constituent members populated 
     /// @return status - true in case of success. 
     void populateGrid(const edm::Event& e, TripletSeederEventState & state) const;
     
     
     /// @brief top-level triplet formation method. 
     /// Will loop over sets of adjacent cells and perform local triplet formation. 
     /// @param state: Event state to take the binned hits from
     /// @param [out] found: Vector of triplets to push the found triplets into 
     void formTriplets(const TripletSeederEventState & state, std::vector<OrderedHitTriplet> & found) const;

     /// @brief Triplet formation method for a given starting cell. 
     /// Will form triplets from hits in the given cell and its neighbours. 
     /// Results will be pushed into the "found" vector, and the "trackUsage"
     /// multiset will count how often a given hit is used to enforce a maximum. 
     /// @param xBin: x-index of the cell to seed triplet formation 
     /// @param yBin: y-index of the cell to seed triplet formation 
     /// @param zBin: z-index of the cell to seed triplet formation 
     /// @param state: event state containing the hits and event information
     /// @param [out] found: The list of triplets to push new candidates into 
     /// @param trackUsage: Counter for the number of triplets a given hit is used for.  
     void formTriplets(int xBin, int yBin, int zBin, 
                       const TripletSeederEventState & state,
                       std::vector<OrderedHitTriplet> & found, 
                       std::unordered_multiset<const BaseTrackerRecHit*> & trackUsage) const;

     /// @brief Triplet formation method for sets of hits to consider. 
     /// Will form triplets comprising one hit from topCands, one from centerCands, and one 
     /// from bottomCands, with the requirement for them to be ordered descending in y. 
     /// @param topCands: Candidates for the top (in global y) hit in the triplet
     /// @param centerCands: Candidates for the middle (in global y) hit in the triplet
     /// @param bottomCands: Candidates for the bottom (in global y) hit in the triplet
     /// @param state: event state containing the hits and event information
     /// @param [out] found: The list of triplets to push new candidates into 
     /// @param trackUsage: Counter for the number of triplets a given hit is used for.  
     void formTriplets(const std::vector<const BaseTrackerRecHit*> & topCands, 
                       const std::vector<const BaseTrackerRecHit*> & centerCands, 
                       const std::vector<const BaseTrackerRecHit*> & bottomCands, 
                       const TripletSeederEventState & state,
                       std::vector<OrderedHitTriplet> & found, 
                       std::unordered_multiset<const BaseTrackerRecHit*> & trackUsage) const;


     /// @brief Top-level method to fit the triplets into trajectorySeeds. 
     /// Will attempt, for each triplet, to check if it satisfies certain quality criteria and fit its 
     /// trajectory using a Kalman filter. 
     /// @param state: Event state, containing the required event info such as field, propagators and updator. 
     /// @param triplets: The set of triplets to fit 
     /// @param output seed collection to push the successfully fit candidates into. 
     void fitTriplets(const TripletSeederEventState & state, const std::vector<OrderedHitTriplet>& triplets, TrajectorySeedCollection & output) const;
     
     /// @brief Method to attempt to fit a single triplet into a trajectorySeed. 
     /// Will attempt to check if it satisfies certain quality criteria and fit its 
     /// trajectory using a Kalman filter. In case of success, will push a candidate into 
     /// the output collection. 
     /// @param state: Event state, containing the required event info such as field, propagators and updator. 
     /// @param triplet: The triplet to fit 
     /// @param output seed collection to push the successfully fit candidates into. 
     /// @return a bool indicating the success of the operation 
     bool fitTriplet(const TripletSeederEventState & state, const OrderedHitTriplet& triplet, TrajectorySeedCollection & output) const;

    /// helper borrowed from SimpleCosmicBONSeeder - possibly refactor into helper class for deployment
    std::pair<GlobalVector, int> pqFromHelixFit(const GlobalPoint &inner,
                                            const GlobalPoint &middle,
                                            const GlobalPoint &outer,
                                            const MagneticField* magfield) const; 

     /// data dependencies 
    edm::EDGetTokenT<VectorHitCollection> vectorHitsToken_;  // vector hit collection
    edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitsToken_;  // strip hits in the outer tracker
    edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> matchedStripHitsToken_;
    edm::EDGetTokenT<SiStripRecHit2DCollection> rPhiHitsToken_;
    edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitsToken_;   // pixel hits

    /// condition dependencies 

    // B-field 
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;

    // geometry
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerToken_;

    // tools 
    const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;

    // config parameters for the seeding grid 
    int nGridBinsX_ = 1;
    int nGridBinsY_ = 1;
    int nGridBinsZ_ = 1;

    double gridXmin_ = -120; 
    double gridXmax_ = 120; 

    double gridYmin_ = -120; 
    double gridYmax_ = 120; 

    double gridZmin_ = -280; 
    double gridZmax_ = 280; 


};