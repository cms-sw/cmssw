#ifndef SimMuon_MCTruth_SeedToTrackProducerBase_h
#define SimMuon_MCTruth_SeedToTrackProducerBase_h

/** \class SeedToTrackProducerBase
 *  
 *  Base class used to produce MuonTracks from seeds
 * 
 *  \author  Hugues Brun
 *  Created:  Tue, 05 Nov 2013 13:42:04 GMT
 *  
 *  Modified to be templated on the seed collection type
 *  to make it compatible with Phase 2 seeds
 *  \author Luca Ferragina (INFN BO), 2024
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

//
// class declaration
//

typedef math::Error<5>::type CovarianceMatrix;

template <typename SeedCollection>
class SeedToTrackProducerBase : public edm::global::EDProducer<> {
public:
  explicit SeedToTrackProducerBase(const edm::ParameterSet &);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const final;

  using SeedType = typename SeedCollection::value_type;
  TrajectoryStateOnSurface seedTransientState(const SeedType &,
                                              const MagneticField &,
                                              const GlobalTrackingGeometry &) const;
  // ----------member data ---------------------------

  edm::EDGetTokenT<SeedCollection> L2seedsTagT_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> L2seedsTagS_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMGFieldToken;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theTrackingGeometryToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theTopoToken;
};

#endif
