#ifndef RecoTracker_TkSeedGenerator_SeedGeneratorFromTTracksEDProducer_H
#define RecoTracker_TkSeedGenerator_SeedGeneratorFromTTracksEDProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilder.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class dso_hidden SeedGeneratorFromTTracksEDProducer : public edm::stream::EDProducer<> {
public:
  SeedGeneratorFromTTracksEDProducer(const edm::ParameterSet& cfg);
  ~SeedGeneratorFromTTracksEDProducer() override {}
  void produce(edm::Event& ev, const edm::EventSetup& es) override;
  void findSeedsOnLayer(const GeometricSearchDet& layer,
			const TrajectoryStateOnSurface& tsosAtIP,
			const Propagator& propagatorAlong,
			const TTTrack< Ref_Phase2TrackerDigi_ >& l1,
			edm::ESHandle<Chi2MeasurementEstimatorBase>& estimatorH,
			unsigned int& numSeedsMade,
			std::unique_ptr<std::vector<TrajectorySeed> >& out) const;

private:
  const edm::ParameterSet theConfig;
  const edm::EDGetTokenT< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > theInputCollectionTag;
  const std::string theEstimatorName;
  const std::string thePropagatorName;
  const edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerTag;

  /// Minimum eta value to activate searching in the TEC
  const double theMinEtaForTEC;

  /// Maximum eta value to activate searching in the TOB
  const double theMaxEtaForTOB;

  const double errorSFHitless;

};
#endif
