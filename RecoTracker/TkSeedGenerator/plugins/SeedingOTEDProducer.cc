
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "MagneticField/Engine/interface/MagneticField.h"

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

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrajectoryStateUpdator;

class SeedingOTEDProducer : public edm::stream::EDProducer<> {
public:
  explicit SeedingOTEDProducer(const edm::ParameterSet&);
  ~SeedingOTEDProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  TrajectorySeedCollection run(edm::Handle<VectorHitCollection>);
  unsigned int checkLayer(unsigned int iidd);
  std::vector<const VectorHit*> collectVHsOnLayer(const edmNew::DetSetVector<VectorHit>&, unsigned int);
  void printVHsOnLayer(const edmNew::DetSetVector<VectorHit>&, unsigned int);
  const TrajectoryStateOnSurface buildInitialTSOS(const VectorHit*) const;
  AlgebraicSymMatrix55 assign44To55(const AlgebraicSymMatrix44&) const;
  std::pair<bool, TrajectoryStateOnSurface> propagateAndUpdate(const TrajectoryStateOnSurface initialTSOS,
                                                               const Propagator&,
                                                               const TrackingRecHit& hit) const;
  float computeGlobalThetaError(const VectorHit* vh, const double sigmaZ_beamSpot) const;
  float computeInverseMomentumError(const VectorHit* vh,
                                    const float globalTheta,
                                    const double sigmaZ_beamSpot,
                                    const double transverseMomentum) const;

  TrajectorySeed createSeed(const TrajectoryStateOnSurface& tsos,
                            const edm::OwnVector<TrackingRecHit>& container,
                            const DetId& id,
                            const Propagator& prop) const;

  struct isInvalid {
    bool operator()(const TrajectoryMeasurement& measurement) {
      return ((measurement.recHit() == nullptr) || !(measurement.recHit()->isValid()) ||
              !((measurement).updatedState().isValid()));
    }
  };

private:
  edm::EDGetTokenT<VectorHitCollection> vhProducerToken_;
  const TrackerTopology* tkTopo_;
  const MeasurementTracker* measurementTracker_;
  std::unique_ptr<LayerMeasurements> layerMeasurements_;
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
  edm::EDPutTokenT<TrajectorySeedCollection> putToken_;
};

SeedingOTEDProducer::SeedingOTEDProducer(edm::ParameterSet const& conf)
    : tkMeasEventToken_(consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent"))),
      topoToken_(esConsumes()),
      propagatorToken_(esConsumes(edm::ESInputTag("", "PropagatorWithMaterial"))),
      magFieldToken_(esConsumes()),
      updatorToken_(esConsumes(edm::ESInputTag("", "KFUpdator"))),
      measurementTrackerToken_(esConsumes()),
      estToken_(esConsumes(edm::ESInputTag("", "Chi2"))) {
  vhProducerToken_ = consumes<VectorHitCollection>(edm::InputTag(conf.getParameter<edm::InputTag>("src")));
  beamSpotToken_ = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotLabel"));
  updatorName_ = conf.getParameter<std::string>("updator");
  putToken_ = produces<TrajectorySeedCollection>();
}

SeedingOTEDProducer::~SeedingOTEDProducer() {}

void SeedingOTEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPhase2VectorHits", "accepted"));
  desc.add<edm::InputTag>("trackerEvent", edm::InputTag("MeasurementTrackerEvent"));
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("updator", std::string("KFUpdator"));
  descriptions.add("SeedingOTEDProducer", desc);
}

void SeedingOTEDProducer::produce(edm::Event& event, const edm::EventSetup& es) {
  tkTopo_ = &es.getData(topoToken_);

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle = es.getHandle(measurementTrackerToken_);
  measurementTracker_ = measurementTrackerHandle.product();

  auto const& measurementTrackerEvent = event.get(tkMeasEventToken_);

  layerMeasurements_ = std::make_unique<LayerMeasurements>(*measurementTrackerHandle, measurementTrackerEvent);

  estimator_ = &es.getData(estToken_);

  propagator_ = &es.getData(propagatorToken_);

  magField_ = &es.getData(magFieldToken_);

  updator_ = &es.getData(updatorToken_);

  beamSpot_ = &event.get(beamSpotToken_);

  // Get the vector hits
  auto vhs = event.getHandle(vhProducerToken_);

  event.emplace(putToken_, run(vhs));
}

TrajectorySeedCollection SeedingOTEDProducer::run(edm::Handle<VectorHitCollection> vHs) {
  TrajectorySeedCollection result;
  //check if all the first three layers have VHs
  std::vector<const VectorHit*> vhSeedsL1 = collectVHsOnLayer(*vHs.product(), 1);
  std::vector<const VectorHit*> vhSeedsL2 = collectVHsOnLayer(*vHs.product(), 2);
  std::vector<const VectorHit*> vhSeedsL3 = collectVHsOnLayer(*vHs.product(), 3);
  if (vhSeedsL1.empty() || vhSeedsL2.empty() || vhSeedsL3.empty()) {
    return result;
  }

  //seeds are built in the L3 of the OT
  const BarrelDetLayer* barrelOTLayer2 = measurementTracker_->geometricSearchTracker()->tobLayers().at(1);

  //the search propag directiondepend on the sign of signZ*signPz, while the building is always the contrary
  Propagator* searchingPropagator = &*propagator_->clone();
  Propagator* buildingPropagator = &*propagator_->clone();
  buildingPropagator->setPropagationDirection(alongMomentum);

  for (const auto& hitL3 : vhSeedsL3) {
    //building a tsos out of a VectorHit
    const TrajectoryStateOnSurface initialTSOS = buildInitialTSOS(hitL3);
    float signZ = copysign(1.0, initialTSOS.globalPosition().z());
    float signPz = copysign(1.0, initialTSOS.globalMomentum().z());

    //set the direction of the propagator
    if (signZ * signPz > 0.0)
      searchingPropagator->setPropagationDirection(oppositeToMomentum);
    else
      searchingPropagator->setPropagationDirection(alongMomentum);

    //find vHits in layer 2
    std::vector<TrajectoryMeasurement> measurementsL2 =
        layerMeasurements_->measurements(*barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator_);

    std::vector<TrajectoryMeasurement>::iterator measurementsL2end =
        std::remove_if(measurementsL2.begin(), measurementsL2.end(), isInvalid());
    measurementsL2.erase(measurementsL2end, measurementsL2.end());

    if (!measurementsL2.empty()) {
      //not sure if building it everytime takes time/memory
      const DetLayer* barrelOTLayer1 = measurementTracker_->geometricSearchTracker()->tobLayers().at(0);

      for (const auto& mL2 : measurementsL2) {
        const TrackingRecHit* hitL2 = mL2.recHit().get();

        //propagate to the L2 and update the TSOS
        std::pair<bool, TrajectoryStateOnSurface> updatedTSOS =
            propagateAndUpdate(initialTSOS, *searchingPropagator, *hitL2);
        if (!updatedTSOS.first)
          continue;

        //searching possible VHs in L1
        std::vector<TrajectoryMeasurement> measurementsL1 =
            layerMeasurements_->measurements(*barrelOTLayer1, updatedTSOS.second, *searchingPropagator, *estimator_);
        std::vector<TrajectoryMeasurement>::iterator measurementsL1end =
            std::remove_if(measurementsL1.begin(), measurementsL1.end(), isInvalid());
        measurementsL1.erase(measurementsL1end, measurementsL1.end());

        if (!measurementsL1.empty()) {
          for (const auto& mL1 : measurementsL1) {
            const TrackingRecHit* hitL1 = mL1.recHit().get();

            //propagate to the L1 and update the TSOS
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL1 =
                propagateAndUpdate(updatedTSOS.second, *searchingPropagator, *hitL1);
            if (!updatedTSOSL1.first)
              continue;

            //building trajectory inside-out
            if (searchingPropagator->propagationDirection() == alongMomentum) {
              buildingPropagator->setPropagationDirection(oppositeToMomentum);
            } else {
              buildingPropagator->setPropagationDirection(alongMomentum);
            }

            updatedTSOSL1.second.rescaleError(100);

            TrajectoryStateOnSurface updatedTSOSL1_final = updator_->update(updatedTSOSL1.second, *hitL1);
            if UNLIKELY (!updatedTSOSL1_final.isValid())
              continue;
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL2_final =
                propagateAndUpdate(updatedTSOSL1_final, *buildingPropagator, *hitL2);
            if (!updatedTSOSL2_final.first)
              continue;
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL3_final =
                propagateAndUpdate(updatedTSOSL2_final.second, *buildingPropagator, *hitL3);
            if (!updatedTSOSL3_final.first)
              continue;

            edm::OwnVector<TrackingRecHit> container;
            container.push_back(hitL1->clone());
            container.push_back(hitL2->clone());
            container.push_back(hitL3->clone());

            TrajectorySeed ts =
                createSeed(updatedTSOSL3_final.second, container, hitL3->geographicalId(), *buildingPropagator);
            result.push_back(ts);
          }
        }
      }
    }
  }
  result.shrink_to_fit();
  return result;
}

unsigned int SeedingOTEDProducer::checkLayer(unsigned int iidd) {
  StripSubdetector strip = StripSubdetector(iidd);
  unsigned int subid = strip.subdetId();
  if (subid == StripSubdetector::TIB || subid == StripSubdetector::TOB) {
    return tkTopo_->layer(iidd);
  }
  return 0;
}

std::vector<const VectorHit*> SeedingOTEDProducer::collectVHsOnLayer(const edmNew::DetSetVector<VectorHit>& input,
                                                                     unsigned int layerNumber) {
  std::vector<const VectorHit*> vHsOnLayer;
  if (!input.empty()) {
    for (const auto& DSViter : input) {
      if (checkLayer(DSViter.id()) == layerNumber) {
        for (const auto& vh : DSViter) {
          vHsOnLayer.push_back(&vh);
        }
      }
    }
  }

  return vHsOnLayer;
}

void SeedingOTEDProducer::printVHsOnLayer(const edmNew::DetSetVector<VectorHit>& input, unsigned int layerNumber) {
  if (!input.empty()) {
    for (const auto& DSViter : input) {
      for (const auto& vh : DSViter) {
        if (checkLayer(DSViter.id()) == layerNumber)
          LogTrace("SeedingOTEDProducer") << " VH in layer " << layerNumber << " >> " << vh;
      }
    }
  } else {
    LogTrace("SeedingOTEDProducer") << " No VHs in layer " << layerNumber << ".";
  }
}

const TrajectoryStateOnSurface SeedingOTEDProducer::buildInitialTSOS(const VectorHit* vHit) const {
  // having fun with theta
  Global3DVector gv(vHit->globalPosition().x(), vHit->globalPosition().y(), vHit->globalPosition().z());
  float theta = gv.theta();
  // gv transform to local (lv)
  const Local3DVector lv(vHit->det()->surface().toLocal(gv));

  //FIXME::charge is fine 1 every two times!!
  GlobalPoint center(0.0, 0.0, 0.0);
  int charge = 1;
  // momentum is a signed quantity in this case
  float mom = vHit->momentum(magField_->inTesla(center).z());
  float xPos = vHit->localPosition().x();
  float yPos = vHit->localPosition().y();
  float dx = vHit->localDirection().x();
  // for dy use second component of the lv renormalized to the z component
  float dy = lv.y() / lv.z();

  // Pz and Dz should have the same sign
  float signPz = copysign(1.0, vHit->globalPosition().z());

  LocalTrajectoryParameters ltpar2(charge / mom, dx, dy, xPos, yPos, signPz);
  AlgebraicSymMatrix55 mat = assign44To55(vHit->covMatrix());
  // set the error on 1/p
  mat[0][0] = pow(computeInverseMomentumError(
                      vHit, theta, beamSpot_->sigmaZ(), vHit->transverseMomentum(magField_->inTesla(center).z())),
                  2);

  //building tsos
  LocalTrajectoryError lterr(mat);
  const TrajectoryStateOnSurface tsos(ltpar2, lterr, vHit->det()->surface(), magField_);

  return tsos;
}

AlgebraicSymMatrix55 SeedingOTEDProducer::assign44To55(const AlgebraicSymMatrix44& mat44) const {
  AlgebraicSymMatrix55 result;
  for (int i = 1; i < 5; i++) {
    for (int j = 1; j < 5; j++) {
      result[i][j] = mat44[i - 1][j - 1];
    }
  }
  return result;
}

std::pair<bool, TrajectoryStateOnSurface> SeedingOTEDProducer::propagateAndUpdate(
    const TrajectoryStateOnSurface initialTSOS, const Propagator& prop, const TrackingRecHit& hit) const {
  TrajectoryStateOnSurface propTSOS = prop.propagate(initialTSOS, hit.det()->surface());
  if UNLIKELY (!propTSOS.isValid())
    return std::make_pair(false, propTSOS);
  TrajectoryStateOnSurface updatedTSOS = updator_->update(propTSOS, hit);
  if UNLIKELY (!updatedTSOS.isValid())
    return std::make_pair(false, updatedTSOS);
  return std::make_pair(true, updatedTSOS);
}

float SeedingOTEDProducer::computeGlobalThetaError(const VectorHit* vh, const double sigmaZ_beamSpot) const {
  double derivative = vh->globalPosition().perp() / vh->globalPosition().mag2();
  double derivative2 = pow(derivative, 2);
  return pow(derivative2 * vh->lowerGlobalPosErr().czz() + derivative2 * pow(sigmaZ_beamSpot, 2), 0.5);
}

float SeedingOTEDProducer::computeInverseMomentumError(const VectorHit* vh,
                                                       const float globalTheta,
                                                       const double sigmaZ_beamSpot,
                                                       const double transverseMomentum) const {
  //for pT > 2GeV, 1/pT has sigma = 1/sqrt(12)
  float varianceInverseTransvMomentum = 1. / 12;
  float derivativeTheta2 = pow(cos(globalTheta) / transverseMomentum, 2);
  float derivativeInverseTransvMomentum2 = pow(sin(globalTheta), 2);
  float thetaError = computeGlobalThetaError(vh, sigmaZ_beamSpot);
  return pow(derivativeTheta2 * pow(thetaError, 2) + derivativeInverseTransvMomentum2 * varianceInverseTransvMomentum,
             0.5);
}

TrajectorySeed SeedingOTEDProducer::createSeed(const TrajectoryStateOnSurface& tsos,
                                               const edm::OwnVector<TrackingRecHit>& container,
                                               const DetId& id,
                                               const Propagator& prop) const {
  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());
  return TrajectorySeed(seedTSOS, container, prop.propagationDirection());
}
DEFINE_FWK_MODULE(SeedingOTEDProducer);
