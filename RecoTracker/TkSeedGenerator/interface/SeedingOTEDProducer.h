//---------------------------------------------------------------------------
// class SeedingOTEDProducer
// author: ebrondol
// date: July, 2016
//---------------------------------------------------------------------------

#ifndef RecoTracker_TkSeedGenerator_SeedingOTEDProducer_h
#define RecoTracker_TkSeedGenerator_SeedingOTEDProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitMomentumHelper.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

class TrajectoryStateUpdator;

class SeedingOTEDProducer : public edm::stream::EDProducer<> {
public:
  explicit SeedingOTEDProducer(const edm::ParameterSet&);
  ~SeedingOTEDProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  TrajectorySeedCollection run(edm::Handle<VectorHitCollectionNew>);
  unsigned int checkLayer(unsigned int iidd);
  std::vector<VectorHit> collectVHsOnLayer(edm::Handle<VectorHitCollectionNew>, unsigned int);
  void printVHsOnLayer(edm::Handle<VectorHitCollectionNew>, unsigned int);
  const TrajectoryStateOnSurface buildInitialTSOS(VectorHit&);
  AlgebraicSymMatrix assign44To55(AlgebraicSymMatrix);
  std::pair<bool, TrajectoryStateOnSurface> propagateAndUpdate(const TrajectoryStateOnSurface initialTSOS,
                                                               const Propagator&,
                                                               const TrackingRecHit& hit);
  float computeGlobalThetaError(const VectorHit& vh, const double sigmaZ_beamSpot);
  float computeInverseMomentumError(VectorHit& vh,
                                    const float globalTheta,
                                    const double sigmaZ_beamSpot,
                                    const double transverseMomentum);

  TrajectorySeed createSeed(const TrajectoryStateOnSurface& tsos,
                            const edm::OwnVector<TrackingRecHit>& container,
                            const DetId& id,
                            const Propagator& prop);

  struct isInvalid {
    bool operator()(const TrajectoryMeasurement& measurement) {
      return (((measurement).recHit() == nullptr) || !((measurement).recHit()->isValid()) ||
              !((measurement).updatedState().isValid()));
    }
  };

private:
  edm::EDGetTokenT<VectorHitCollectionNew> vhProducerToken_;
  const TrackerTopology* tkTopo_;
  const MeasurementTracker* measurementTracker_;
  const LayerMeasurements* layerMeasurements_;
  const MeasurementEstimator* estimator_;
  const Propagator* propagator_;
  const MagneticField* magField_;
  const TrajectoryStateUpdator* updator_;
  const edm::EDGetTokenT<MeasurementTrackerEvent> tkMeasEventToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const reco::BeamSpot* beamSpot_;
  std::string updatorName_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  edm::ESGetToken<TrajectoryStateUpdator, TrackingComponentsRecord> updatorToken_;
  edm::ESGetToken<MeasurementTracker, CkfComponentsRecord> measurementTrackerToken_;
  edm::ESGetToken<Chi2MeasurementEstimatorBase, TrackingComponentsRecord> estToken_;

};

#endif
