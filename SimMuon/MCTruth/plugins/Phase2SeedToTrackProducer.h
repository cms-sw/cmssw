#ifndef SimMuon_MCTruth_Phase2SeedToTrackProducer_h
#define SimMuon_MCTruth_Phase2SeedToTrackProducer_h

/** \class Phase2SeedToTrackProducer
 * 
 *  Phase-2 implementation of the SeedToTrackProducerModule.
 *  Baseline behaviour is the same, with the difference that
 *  a collection of L2MuonTrajectorySeeds is expected as input
 * 
 *  \author Luca Ferragina (INFN BO), Carlo Battilana (INFN BO), 2024
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

//
// class declaration
//

typedef math::Error<5>::type CovarianceMatrix;

class Phase2SeedToTrackProducer : public edm::global::EDProducer<> {
public:
  explicit Phase2SeedToTrackProducer(const edm::ParameterSet &);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const final;
  TrajectoryStateOnSurface seedTransientState(const L2MuonTrajectorySeed &,
                                              const MagneticField &,
                                              const GlobalTrackingGeometry &) const;
  // ----------member data ---------------------------

  edm::EDGetTokenT<L2MuonTrajectorySeedCollection> L2seedsTagT_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> L2seedsTagS_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMGFieldToken;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theTrackingGeometryToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theTopoToken;
};

#endif
