# include <Rtypes.h>
#include <RtypesCore.h>
#include <TAttLine.h>
#include <TAttMarker.h>
#include <TEllipse.h>
#include <cmath>
#include <memory>
#include <set>
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicGridTripletSeeder.h"

//


CosmicGridTripletSeeder::CosmicGridTripletSeeder(const edm::ParameterSet& iConfig)
    : vectorHitsToken_(mayConsume<VectorHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vectorHits"))),
      otRecHitsToken_(mayConsume<Phase2TrackerRecHit1DCollectionNew>(iConfig.getUntrackedParameter<edm::InputTag>("OTRecHits"))),
      matchedStripHitsToken_(mayConsume<SiStripMatchedRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("matchedStripHits"))),
      rPhiHitsToken_(mayConsume<SiStripRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("rPhiHits"))),
      pixelRecHitsToken_(consumes(iConfig.getUntrackedParameter<edm::InputTag>("PixelRecHits"))),
      magfieldToken_(esConsumes(iConfig.getParameter<edm::ESInputTag>("MagneticFieldRecord"))),
      trackerToken_(esConsumes()),
      ttrhBuilderToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("TTRHBuilder")))),
      nGridBinsX_(iConfig.getParameter<int>("nGridBinsX")),
      nGridBinsY_(iConfig.getParameter<int>("nGridBinsY")),
      nGridBinsZ_(iConfig.getParameter<int>("nGridBinsZ")),
      gridXmin_(iConfig.getParameter<double>("gridXmin")),
      gridXmax_(iConfig.getParameter<double>("gridXmax")),
      gridYmin_(iConfig.getParameter<double>("gridYmin")),
      gridYmax_(iConfig.getParameter<double>("gridYmax")),
      gridZmin_(iConfig.getParameter<double>("gridZmin")),
      gridZmax_(iConfig.getParameter<double>("gridZmax")){
  produces<TrajectorySeedCollection>();
  produces<edm::OwnVector<TrackingRecHit>>();

}

void CosmicGridTripletSeeder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("vectorHits", edm::InputTag("siPhase2VectorHits:accepted"));
  desc.addUntracked<edm::InputTag>("OTRecHits", edm::InputTag("siPhase2RecHits"));
  desc.addUntracked<edm::InputTag>("matchedStripHits", edm::InputTag("siStripMatchedRecHits","matchedRecHit"));
  desc.addUntracked<edm::InputTag>("rPhiHits", edm::InputTag("siStripMatchedRecHits","rphiRecHit"));
  desc.addUntracked<edm::InputTag>("PixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag("", ""));
  desc.add<int>("nGridBinsX",1);
  desc.add<int>("nGridBinsY",1);
  desc.add<int>("nGridBinsZ",1);
  desc.add<double>("gridXmin",-120);
  desc.add<double>("gridXmax",120);
  desc.add<double>("gridYmin",-120);
  desc.add<double>("gridYmax",120);
  desc.add<double>("gridZmin",-280);
  desc.add<double>("gridZmax",280);
  descriptions.addWithDefaultLabel(desc);
}

std::pair<GlobalVector, int> CosmicGridTripletSeeder::pqFromHelixFit(const GlobalPoint& inner,
                                                                     const GlobalPoint& middle,
                                                                     const GlobalPoint& outer,
                                                                     const MagneticField* magfield) const {
  FastHelix helix(inner, middle, outer, magfield->nominalValue(), magfield);
  FastCircle theCircle(inner, middle, outer);
  double rho = theCircle.rho();
  GlobalVector tesla = magfield->inTesla(middle);
  double pt = 0.01 * rho * (0.3 * tesla.z());
  double dx1 = outer.x() - theCircle.x0();
  double dy1 = outer.y() - theCircle.y0();
  double py = pt * dx1 / rho, px = -pt * dy1 / rho;
  if (px * (middle.x() - outer.x()) + py * (middle.y() - outer.y()) < 0.) {
    px *= -1.;
    py *= -1.;
  }
  double dz = inner.z() - outer.z();
  double sinphi = (dx1 * (inner.y() - theCircle.y0()) - dy1 * (inner.x() - theCircle.x0())) / (rho * rho);
  double dphi = std::abs(std::asin(sinphi));
  double pz = pt * dz / (dphi * rho);

  int myq = ((theCircle.x0() * py - theCircle.y0() * px) / tesla.z()) > 0. ? +1 : -1;

  return std::make_pair(GlobalVector(px, py, pz), myq);

}

CosmicGridTripletSeeder::TripletSeederEventState CosmicGridTripletSeeder::initEventState(const edm::EventSetup& c) const {
  auto magfield = &c.getData(magfieldToken_);
  auto tracker = &c.getData(trackerToken_);
  auto cloner = dynamic_cast<TkTransientTrackingRecHitBuilder const&>(c.getData(ttrhBuilderToken_)).cloner();

  return TripletSeederEventState{CartesianSeedingGrid{nGridBinsX_, gridXmin_, gridXmax_, nGridBinsY_, gridYmin_, gridYmax_, nGridBinsZ_, gridZmin_, gridZmax_},
                                 {},
                                 magfield,
                                 tracker,
                                 cloner,
                                 std::make_unique<KFUpdator>(),
                                 std::make_unique<PropagatorWithMaterial>(alongMomentum, 0.1057, magfield),
                                 std::make_unique<PropagatorWithMaterial>(oppositeToMomentum, 0.1057, magfield)};
}

void CosmicGridTripletSeeder::produce(edm::Event& e, const edm::EventSetup& c) {
  // setup the event state object
  TripletSeederEventState state = initEventState(c);

  // populate the seeding grid
  populateGrid(e,state);

  // now form triplets
  std::vector<OrderedHitTriplet> triplets;
  formTriplets(state, triplets);

  // book the trajectory seed collection
  auto output = std::make_unique<TrajectorySeedCollection>();
  auto outtriplets = std::make_unique<edm::OwnVector<TrackingRecHit>>();

  // and fit the triplets
  fitTriplets(state, triplets, *output);

  for (auto & trip : triplets){
    outtriplets->push_back(trip.inner()->cloneHit()); 
    outtriplets->push_back(trip.middle()->cloneHit()); 
    outtriplets->push_back(trip.outer()->cloneHit()); 
  }

  // put our output on the store
  e.put(std::move(output));
  e.put(std::move(outtriplets));
}

void CosmicGridTripletSeeder::populateGrid(const edm::Event& iEvent, CosmicGridTripletSeeder::TripletSeederEventState & state) const {
  edm::Handle<VectorHitCollection> vectorHits; 
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> otHitCollection; 
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedStripHits; 
  edm::Handle<SiStripRecHit2DCollection> rPhiStripHits; 

  /// Treat the hit collections as optional, so that we can
  /// run with both the Run-3 and the Phase-2 detectors. 
  bool hasOT = iEvent.getByToken(otRecHitsToken_, otHitCollection );
  bool hasVec = iEvent.getByToken(vectorHitsToken_, vectorHits );
  bool hasMatchedStrips = iEvent.getByToken(matchedStripHitsToken_, matchedStripHits );
  bool hasRphiStrips = iEvent.getByToken(rPhiHitsToken_, rPhiStripHits );

  /// Pixels should be around in all detector geometries. 
  const SiPixelRecHitCollection & pixelHitCollection = iEvent.get(pixelRecHitsToken_);

  std::set<const TrackingRecHit*> recHitsSeen{}; 
  std::vector<const VectorHit*> vhSeen{}; 

  /// step 1: Add the vector hits, and remember all raw hits associated to them 
  if (hasVec){
    LogDebug("CosmicGridTrupletSeeder")<< "have Vec hits "; 
    for (auto detSet : *vectorHits){
        for (const VectorHit & vh : detSet){
          // Phase-2 endcap vector hits currently have directional information
          // that can not be used directly - hence we skip these. 
          if (state.tracker->idToDet(vh.geographicalId())->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC){
            continue; 
          }
          state.grid.addHit(&vh); 
          vhSeen.push_back(&vh); 
        }
    }
  }

  // Step 2 for phase-2 detector

  // Add remaining OT hits, excluding those already on vector hits 
  if (hasOT){
    LogDebug("CosmicGridTrupletSeeder")<< "have OT hits "; 
    for (auto  ds : *otHitCollection){
        for (const auto & otHit : ds){
          bool unique = true; 
          for (const auto* vh : vhSeen){
              if (vh->sharesInput(&otHit,TrackingRecHit::some)){
                  unique = false; 
                  state.vhConstituents[vh].push_back(&otHit);
                  break; 
              }
          }
          if (!unique){
              continue; 
          }
          state.grid.addHit(&otHit); 
          recHitsSeen.insert(&otHit); 
        }
    }
  }

  // Step 2 for a phase-1 detector (two hit collections)

  // phase-1 matched strip hits
  if (hasMatchedStrips){
    LogDebug("CosmicGridTrupletSeeder")<< "have matched hits "; 
    for (auto  ds : *matchedStripHits){
      for (const auto & stripHit : ds){
        bool unique = true; 
        for (const auto* vh : vhSeen){
            if (vh->sharesInput(&stripHit,TrackingRecHit::some)){
                unique = false; 
                state.vhConstituents[vh].push_back(&stripHit);
                break; 
            }
        }
        if (!unique){
            continue; 
        }
        state.grid.addHit(&stripHit); 
        recHitsSeen.insert(&stripHit); 
      }
    }
  } 

  // phase-1 rphi strip hits 
  if (hasRphiStrips){
    LogDebug("CosmicGridTrupletSeeder")<< "have rphi hits "; 
    for (auto  ds : *rPhiStripHits){
      for (const auto & stripHit : ds){
        bool unique = true; 
        for (const auto* vh : vhSeen){
            if (vh->sharesInput(&stripHit,TrackingRecHit::some)){
                unique = false; 
                state.vhConstituents[vh].push_back(&stripHit);
                break; 
            }
        }
        if (!unique){
            continue; 
        }
        state.grid.addHit(&stripHit); 
        recHitsSeen.insert(&stripHit); 
      }
    }
  }

  /// step 3: Add pixel hits if desired  
  for (auto  ds : pixelHitCollection){
      for (const auto & pix : ds){
        state.grid.addHit(&pix); 
      }
  }

  // now sort all bins of the grid by ascending global y.
  state.grid.sort(); 

}

/// using the seeding grid, form triplets
void CosmicGridTripletSeeder::formTriplets(const CosmicGridTripletSeeder::TripletSeederEventState & state, std::vector<OrderedHitTriplet> & found) const {
    std::unordered_multiset<const BaseTrackerRecHit*> trackUsage; 
    // loop downwards over the starting y bin, top to bottom 
    for (int yBin = state.grid.nBinsY()-1; yBin >=0 ; --yBin){
      // loop rectangularily over the x-z grid 
      for (int xBin = 0; xBin < state.grid.nBinsX(); ++xBin){
        for (int zBin = 0; zBin < state.grid.nBinsZ(); ++zBin){
          const auto & topCandidates = state.grid.getHits(xBin,yBin,zBin); 
          if (topCandidates.empty()) continue;         
          formTriplets(xBin,yBin,zBin,state,found,trackUsage); 
        }
      }
    }
}

void CosmicGridTripletSeeder::formTriplets(int xBin,
                                           int yBin,
                                           int zBin,
                                           const TripletSeederEventState& state,
                                           std::vector<OrderedHitTriplet>& found,
                                           std::unordered_multiset<const BaseTrackerRecHit*>& trackUsage) const {
  const CartesianSeedingGrid& grid = state.grid;

  const auto& topHits = grid.getHits(xBin, yBin, zBin);

  std::vector<const BaseTrackerRecHit *> hitCands; 
  hitCands.insert(hitCands.end(),topHits.begin(), topHits.end() ); 

  for (int dxCenter = -1; dxCenter < 2; ++dxCenter) {
    if (xBin + dxCenter < 0 || xBin + dxCenter >= grid.nBinsX())
      continue;
    for (int dzCenter = -1; dzCenter < 2; ++dzCenter) {
      if (zBin + dzCenter < 0 || zBin + dzCenter >= grid.nBinsZ())
        continue;
      // formTriplets(topHits, topHits, topHits, state, found, trackUsage);
      if (yBin > 0){
        const auto& middleHits = grid.getHits(xBin + dxCenter, yBin - 1, zBin + dzCenter);
        hitCands.insert(hitCands.end(),middleHits.begin(), middleHits.end() ); 
        // // formTriplets(topHits, topHits, middleHits, state, found, trackUsage);
        // // formTriplets(topHits, middleHits, middleHits, state, found, trackUsage);
        // for (int dxBottom = -1; dxBottom < 2; ++dxBottom) {
        //   if (xBin + dxCenter + dxBottom < 0 || xBin + dxCenter + dxBottom >= grid.nBinsX())
        //     continue;
        //   if ((dxCenter < 0 && dxBottom > 0) || (dxCenter > 0 && dxBottom < 0))
        //     continue;
        //   for (int dzBottom = -1; dzBottom < 2; ++dzBottom) {
        //     if (zBin + dzCenter + dzBottom < 0 || zBin + dzCenter + dzBottom >= grid.nBinsZ())
        //       continue;
        //     if ((dzCenter < 0 && dzBottom > 0) || (dzCenter > 0 && dzBottom < 0))
        //       continue;
        //     if (yBin > 1){
        //       const auto& bottomHits = grid.getHits(xBin + dxCenter + dxBottom, yBin - 2, zBin + dzCenter + dzBottom);
        //       formTriplets(topHits, middleHits, bottomHits, state, found, trackUsage);
        //     }
        //   }
        // }
      }
    }
  }
  std::sort(hitCands.begin(),hitCands.end(),[](const BaseTrackerRecHit* h1, const BaseTrackerRecHit* h2){return h1->globalPosition().y() < h2->globalPosition().y(); });
  formTriplets(hitCands, hitCands, hitCands, state, found, trackUsage);
}

     /// get the triplets for one particular cell combination
void CosmicGridTripletSeeder::formTriplets(const std::vector<const BaseTrackerRecHit*> & topCands, 
                const std::vector<const BaseTrackerRecHit*> & centerCands, 
                const std::vector<const BaseTrackerRecHit*> & bottomCands, 
                const TripletSeederEventState & state,
                std::vector<OrderedHitTriplet> & found, 
                std::unordered_multiset<const BaseTrackerRecHit*> & trackUsage) const{

  for (const BaseTrackerRecHit* top : topCands){
    if (trackUsage.count(top) > 12) continue; 
    const auto & gTop = top->globalPosition(); 
    for (const BaseTrackerRecHit* center: centerCands){
      const auto & gCenter = center->globalPosition(); 
      if (trackUsage.count(center) > 12) continue; 
      for (const BaseTrackerRecHit* bottom: bottomCands){
        const auto & gBottom = bottom->globalPosition(); 
        if (trackUsage.count(bottom) > 12) continue; 
        if (center == top || top == bottom || center == bottom) continue; 
        if (top->sameDetModule(*bottom) || top->sameDetModule(*center) || center->sameDetModule(*bottom)) continue; 
        if (gCenter.y() > gTop.y()) continue;
        if (gBottom.y() > gCenter.y()) continue;
        // veto "zigzag" in z 
        if ((gTop.z() - gCenter.z()) * (gCenter.z() - gBottom.z())  < 0 && std::abs(gTop.z() - gBottom.z() > 1.0)) continue;
        // veto "zigzag" in x 
        if ((gTop.x() - gCenter.x()) * (gCenter.x() - gBottom.x())  < 0 && std::abs(gTop.x() - gBottom.x() > 1.0)) continue;
        const VectorHit* cenVH = dynamic_cast<const VectorHit*>(center); 
        if (cenVH){
          double dydx = 0.5 * ((gTop.y() - gCenter.y())/(gTop.x() - gCenter.x()) + (gCenter.y() - gBottom.y())/(gCenter.x() - gBottom.x())); 
          double dydx_vh = cenVH->globalDirectionVH().y() / cenVH->globalDirectionVH().x();
          LogDebug("CosmicGridTripletSeeder") << " dydx from trip "<<dydx<<" and from vh "<<dydx_vh;
        }

        OrderedHitTriplet ps (top, center, bottom);
        trackUsage.insert(top); 
        trackUsage.insert(center); 
        trackUsage.insert(bottom); 
        found.push_back(ps); 
      }
    }
  }
}
/// fit the triplets with kalman into trajectory seeds
void CosmicGridTripletSeeder::fitTriplets(const CosmicGridTripletSeeder::TripletSeederEventState & state, const std::vector<OrderedHitTriplet>& triplets, TrajectorySeedCollection & output) const {
    for (const OrderedHitTriplet & trip : triplets ) {
      fitTriplet(state, trip, output); 
    }
}

/// fit of a single triplet into a trajectory seed 
bool CosmicGridTripletSeeder::fitTriplet(const CosmicGridTripletSeeder::TripletSeederEventState & state, const OrderedHitTriplet& triplet, TrajectorySeedCollection & output) const {
  typedef TrajectoryStateOnSurface TSOS;

    OrderedHitTriplet trip = triplet;  

    GlobalPoint inner =
        state.tracker->idToDet((*(trip.inner())).geographicalId())->surface().toGlobal((*(trip.inner())).localPosition());

    GlobalPoint middle =
        state.tracker->idToDet((*(trip.middle())).geographicalId())->surface().toGlobal((*(trip.middle())).localPosition());

    GlobalPoint outer =
        state.tracker->idToDet((*(trip.outer())).geographicalId())->surface().toGlobal((*(trip.outer())).localPosition());
    if ((outer.y() - inner.y()) * outer.y() < 0) {
      std::swap(inner, outer);
      trip = OrderedHitTriplet(trip.outer(), trip.middle(), trip.inner());
    }
    // First use FastHelix out of the box
    std::pair<GlobalVector, int> pq = pqFromHelixFit(inner, middle, outer, state.magfield);
    GlobalVector gv = pq.first;
    float ch = pq.second;
    float Mom = sqrt(gv.x() * gv.x() + gv.y() * gv.y() + gv.z() * gv.z());
    if (Mom > 1000000 || edm::isNotFinite(Mom)) {
      return false;
    }
    if (gv.perp() < 0.5) {
      return false;
    }
    const Propagator *propagator = nullptr;
    if ((outer.y() - inner.y()) > 0) {
      propagator = state.thePropagatorAl.get();
    } else {
      gv = -1 * gv;
      ch = -1. * ch;
      propagator = state.thePropagatorOp.get();
    }
    if ((gv.z() * (outer.z() - inner.z()) > 0) && (fabs(outer.z() - inner.z()) > 5) && (fabs(gv.z()) > .01)) {
    }

    edm::OwnVector<TrackingRecHit> hits;
    std::vector<const BaseTrackerRecHit*> seedHits;
    for (const BaseTrackerRecHit* hit : {trip.outer(), trip.middle(), trip.inner()}){
      const VectorHit* vh = dynamic_cast<const VectorHit*>(hit);
      if (vh){
        auto found = state.vhConstituents.find(vh);
        if (found != state.vhConstituents.end()){
          for (auto & component : found->second){
            seedHits.push_back(component);
          }
        }
        else{
            edm::LogWarning("CosmicGridTripletSeeder")<< "Did not find any constitutents for a vector hit - are our inputs consistent?";
            seedHits.push_back(hit);
        }
      }
      else{
        seedHits.push_back(hit);
      }
    }
    std::sort(seedHits.begin(),seedHits.end(),[&](const BaseTrackerRecHit* h1, const BaseTrackerRecHit* h2){
      if (outer.y() > 0 ){
        return h1->globalPosition().y() > h2->globalPosition().y();
      }
      else{
        return h1->globalPosition().y() < h2->globalPosition().y();
      }
    });
    outer = seedHits.front()->globalPosition(); 
    GlobalTrajectoryParameters Gtp(outer, gv, int(ch), state.magfield);
    FreeTrajectoryState CosmicSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
    CosmicSeed.rescaleError(100);
    TSOS propagated, updated;
    bool fail = false;
    for (size_t ih = 0; ih < seedHits.size(); ++ih) {
      if (ih == 0) {
        propagated = propagator->propagate(CosmicSeed, state.tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
      } else {
        propagated = propagator->propagate(updated, state.tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
      }
      if (!propagated.isValid()) {
        fail = true;
        break;
      } else {
      }
      SeedingHitSet::ConstRecHitPointer tthp = seedHits[ih];
      auto newtth = static_cast<SeedingHitSet::RecHitPointer>(state.cloner(*tthp, propagated));
      updated = state.theUpdator->update(propagated, *newtth);
      hits.push_back(newtth);
      if (!updated.isValid()) {
        fail = true;
        break;
      } else {
      }
    }
    if (!fail && updated.isValid() && (updated.globalMomentum().perp() < 2.5)) {
      fail = true;
    }
    if (!fail && updated.isValid() && (updated.globalMomentum().mag() < 2.5)) {
      fail = true;
    }
    if (fail) return false; 
    // if (!fail) {
        // if (seedVerbosity_ > 2) {
        //   std::cout << "Processing triplet "  << ", rescale error by " << rescaleError_
        //             << ": state BEFORE rescaling " << updated;
        //   std::cout << "    Cartesian error (X,P) before rescaling= \n"
        //             << updated.cartesianError().matrix() << std::endl;
        // }
        // updated.rescaleError(100);
      // }
    //   if (true) {
    //   std::cout << "Processed  triplet "  << ": success (saved as #" << output.size() << ") : " << inner << " + "
    //             << middle << " + " << outer << std::endl;
    //   std::cout << "    pt = " << updated.globalMomentum().perp() << "    eta = " << updated.globalMomentum().eta()
    //             << "    phi = " << updated.globalMomentum().phi() << "    ch = " << updated.charge() << std::endl;
    //   if (true) {
    //     std::cout << "    State:" << updated;
    //   } else {
    //     std::cout << "    X  = " << updated.globalPosition() << ", P = " << updated.globalMomentum() << std::endl;
    //   }
    //   std::cout << "    Cartesian error (X,P) = \n" << updated.cartesianError().matrix() << std::endl;
    // }   
    PTrajectoryStateOnDet const &PTraj = trajectoryStateTransform::persistentState(
        updated, hits.back().geographicalId().rawId());
    
    output.push_back(TrajectorySeed(PTraj, hits, ((outer.y() - inner.y() > 0) ? alongMomentum : oppositeToMomentum)));


    if (output.size() > size_t(500)) {
      output.clear();
      edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
      return false;
    }

    return true;

}