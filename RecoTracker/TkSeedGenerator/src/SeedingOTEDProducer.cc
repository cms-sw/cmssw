#include "RecoTracker/TkSeedGenerator/interface/SeedingOTEDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MeasurementDet/interface/TrajectoryMeasurementGroup.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SeedingOTEDProducer::SeedingOTEDProducer(edm::ParameterSet const& conf)
    : theUpdator(nullptr),
      tkMeasEventToken(consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent"))) {
  vhProducerToken = consumes<VectorHitCollectionNew>(edm::InputTag(conf.getParameter<edm::InputTag>("src")));
  beamSpotToken = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotLabel"));
  updatorName = conf.getParameter<std::string>("updator");
  produces<TrajectorySeedCollection>();
}

SeedingOTEDProducer::~SeedingOTEDProducer() {}

void SeedingOTEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPhase2VectorHits", "vectorHitsAccepted"));
  desc.add<edm::InputTag>("trackerEvent", edm::InputTag("MeasurementTrackerEvent"));
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("updator", std::string("KFUpdator"));
  descriptions.add("SeedingOTEDProducer", desc);
}

void SeedingOTEDProducer::produce(edm::Event& event, const edm::EventSetup& es) {
  std::unique_ptr<TrajectorySeedCollection> seedsWithVHs(new TrajectorySeedCollection());

  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  tkTopo = tTopoHandle.product();

  edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(measurementTrackerHandle);
  measurementTracker = measurementTrackerHandle.product();
  edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
  event.getByToken(tkMeasEventToken, measurementTrackerEvent);

  layerMeasurements = new LayerMeasurements(*measurementTrackerHandle, *measurementTrackerEvent);

  edm::ESHandle<Chi2MeasurementEstimatorBase> est;
  es.get<TrackingComponentsRecord>().get("Chi2", est);
  estimator = est.product();

  edm::ESHandle<Propagator> prop;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", prop);
  propagator = prop.product();

  edm::ESHandle<MagneticField> magFieldHandle;
  es.get<IdealMagneticFieldRecord>().get(magFieldHandle);
  magField = magFieldHandle.product();

  edm::ESHandle<TrajectoryStateUpdator> updatorHandle;
  es.get<TrackingComponentsRecord>().get(updatorName, updatorHandle);
  theUpdator = updatorHandle.product();

  edm::Handle<reco::BeamSpot> beamSpotH;
  event.getByToken(beamSpotToken, beamSpotH);
  if (beamSpotH.isValid()) {
    beamSpot = beamSpotH.product();
  }

  // Get the vector hits
  edm::Handle<VectorHitCollectionNew> vhs;
  event.getByToken(vhProducerToken, vhs);
  /*
  edm::ESHandle< ClusterParameterEstimator<Phase2TrackerCluster1D> > parameterestimator;
  es.get<TkStripCPERecord>().get(cpe, parameterestimator); 
  const Phase2StripCPEGeometric & cpeOT(*parameterestimator);
*/
  TrajectorySeedCollection const& tempSeeds = run(vhs);
  for (TrajectorySeedCollection::const_iterator qIt = tempSeeds.begin(); qIt < tempSeeds.end(); ++qIt) {
    seedsWithVHs->push_back(*qIt);
  }

  seedsWithVHs->shrink_to_fit();
  event.put(std::move(seedsWithVHs));
}

TrajectorySeedCollection SeedingOTEDProducer::run(edm::Handle<VectorHitCollectionNew> VHs) {
  TrajectorySeedCollection result;

  //check if all the first three layers have VHs
  std::vector<VectorHit> VHseedsL1 = collectVHsOnLayer(VHs, 1);
  std::vector<VectorHit> VHseedsL2 = collectVHsOnLayer(VHs, 2);
  std::vector<VectorHit> VHseedsL3 = collectVHsOnLayer(VHs, 3);
  if (VHseedsL1.empty() || VHseedsL2.empty() || VHseedsL3.empty()) {
    return result;
  }

  //seeds are built in the L3 of the OT
  const BarrelDetLayer* barrelOTLayer2 = measurementTracker->geometricSearchTracker()->tobLayers().at(1);

  //the search propag directiondepend on the sign of signZ*signPz, while the building is always the contrary
  Propagator* searchingPropagator = &*propagator->clone();
  Propagator* buildingPropagator = &*propagator->clone();
  buildingPropagator->setPropagationDirection(alongMomentum);

  for (auto hitL3 : VHseedsL3) {
    //building a tsos out of a VectorHit
    const TrajectoryStateOnSurface initialTSOS = buildInitialTSOS(hitL3);
    float signZ = copysign(1.0, initialTSOS.globalPosition().z());
    float signPz = copysign(1.0, initialTSOS.globalMomentum().z());

    //set the direction of the propagator
    if (signZ * signPz > 0.0)
      searchingPropagator->setPropagationDirection(oppositeToMomentum);
    if (signZ * signPz < 0.0)
      searchingPropagator->setPropagationDirection(alongMomentum);

    //find vHits in layer 2
    std::vector<TrajectoryMeasurement> measurementsL2 =
        layerMeasurements->measurements(*barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator);

    //other options
    //LayerMeasurements::SimpleHitContainer hits;
    //layerMeasurements->recHits(hits, *barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator);
    //auto && measurementsL2G = layerMeasurements->groupedMeasurements(*barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator);

    std::vector<TrajectoryMeasurement>::iterator measurementsL2end =
        std::remove_if(measurementsL2.begin(), measurementsL2.end(), isInvalid());
    measurementsL2.erase(measurementsL2end, measurementsL2.end());

    if (!measurementsL2.empty()) {
      //not sure if building it everytime takes time/memory
      const DetLayer* barrelOTLayer1 = measurementTracker->geometricSearchTracker()->tobLayers().at(0);

      for (auto mL2 : measurementsL2) {
        const TrackingRecHit* hitL2 = mL2.recHit().get();

        //propagate to the L2 and update the TSOS
        std::pair<bool, TrajectoryStateOnSurface> updatedTSOS =
            propagateAndUpdate(initialTSOS, *searchingPropagator, *hitL2);
        if (!updatedTSOS.first)
          continue;

        //searching possible VHs in L1
        std::vector<TrajectoryMeasurement> measurementsL1 =
            layerMeasurements->measurements(*barrelOTLayer1, updatedTSOS.second, *searchingPropagator, *estimator);
        std::vector<TrajectoryMeasurement>::iterator measurementsL1end =
            std::remove_if(measurementsL1.begin(), measurementsL1.end(), isInvalid());
        measurementsL1.erase(measurementsL1end, measurementsL1.end());

        if (!measurementsL1.empty()) {
          for (auto mL1 : measurementsL1) {
            const TrackingRecHit* hitL1 = mL1.recHit().get();

            //propagate to the L1 and update the TSOS
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL1 =
                propagateAndUpdate(updatedTSOS.second, *searchingPropagator, *hitL1);
            if (!updatedTSOSL1.first)
              continue;

            edm::OwnVector<TrackingRecHit> container;
            container.push_back(hitL1->clone());
            container.push_back(hitL2->clone());
            container.push_back(hitL3.clone());

            //building trajectory inside-out
            if (searchingPropagator->propagationDirection() == alongMomentum) {
              buildingPropagator->setPropagationDirection(oppositeToMomentum);
            } else if (searchingPropagator->propagationDirection() == oppositeToMomentum) {
              buildingPropagator->setPropagationDirection(alongMomentum);
            }

            updatedTSOSL1.second.rescaleError(100);

            TrajectoryStateOnSurface updatedTSOSL1_final = theUpdator->update(updatedTSOSL1.second, *hitL1);
            if
              UNLIKELY(!updatedTSOSL1_final.isValid()) continue;
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL2_final =
                propagateAndUpdate(updatedTSOSL1_final, *buildingPropagator, *hitL2);
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL3_final =
                propagateAndUpdate(updatedTSOSL2_final.second, *buildingPropagator, hitL3);
            TrajectorySeed ts =
                createSeed(updatedTSOSL3_final.second, container, hitL3.geographicalId(), *buildingPropagator);
            result.push_back(ts);
          }
        }
      }
    }
  }

  return result;
}

unsigned int SeedingOTEDProducer::checkLayer(unsigned int iidd) {
  StripSubdetector strip = StripSubdetector(iidd);
  unsigned int subid = strip.subdetId();
  if (subid == StripSubdetector::TIB || subid == StripSubdetector::TOB) {
    return tkTopo->layer(iidd);
  }
  return 0;
}

std::vector<VectorHit> SeedingOTEDProducer::collectVHsOnLayer(edm::Handle<VectorHitCollectionNew> VHs,
                                                              unsigned int layerNumber) {
  const VectorHitCollectionNew& input = *VHs;
  std::vector<VectorHit> VHsOnLayer;
  if (!input.empty()) {
    for (auto DSViter : input) {
      if (checkLayer(DSViter.id()) == layerNumber) {
        for (auto vh : DSViter) {
          VHsOnLayer.push_back(vh);
        }
      }
    }
  }

  return VHsOnLayer;
}

void SeedingOTEDProducer::printVHsOnLayer(edm::Handle<VectorHitCollectionNew> VHs, unsigned int layerNumber) {
  const VectorHitCollectionNew& input = *VHs;
  if (!input.empty()) {
    for (auto DSViter : input) {
      for (auto vh : DSViter) {
        if (checkLayer(DSViter.id()) == layerNumber)
          std::cout << " VH in layer " << layerNumber << " >> " << vh << std::endl;
      }
    }
  } else {
    std::cout << " No VHs in layer " << layerNumber << "." << std::endl;
  }
}

const TrajectoryStateOnSurface SeedingOTEDProducer::buildInitialTSOS(VectorHit& vHit) {
  // having fun with theta
  Global3DVector gv(vHit.globalPosition().x(), vHit.globalPosition().y(), vHit.globalPosition().z());
  float theta = gv.theta();
  // gv transform to local (lv)
  const Local3DVector lv(vHit.det()->surface().toLocal(gv));

  //FIXME::charge is fine 1 every two times!!
  int charge = 1;
  float p = vHit.momentum(magField);
  float x = vHit.localPosition().x();
  float y = vHit.localPosition().y();
  float dx = vHit.localDirection().x();
  // for dy use second component of the lv renormalized to the z component
  float dy = lv.y() / lv.z();

  // Pz and Dz should have the same sign
  float signPz = copysign(1.0, vHit.globalPosition().z());

  LocalTrajectoryParameters ltpar2(charge / p, dx, dy, x, y, signPz);
  AlgebraicSymMatrix mat = assign44To55(vHit.parametersError());
  // set the error on 1/p
  mat[0][0] = pow(computeInverseMomentumError(vHit, theta, magField, beamSpot->sigmaZ()), 2);

  //building tsos
  LocalTrajectoryError lterr(asSMatrix<5>(mat));
  const TrajectoryStateOnSurface tsos(ltpar2, lterr, vHit.det()->surface(), magField);

  return tsos;
}

AlgebraicSymMatrix SeedingOTEDProducer::assign44To55(AlgebraicSymMatrix mat44) {
  if (mat44.num_row() != 4 || mat44.num_col() != 4)
    assert("Wrong dimension! This should be a 4x4 matrix!");

  AlgebraicSymMatrix result(5, 0);
  for (int i = 1; i < 5; i++) {
    for (int j = 1; j < 5; j++) {
      result[i][j] = mat44[i - 1][j - 1];
    }
  }
  return result;
}

std::pair<bool, TrajectoryStateOnSurface> SeedingOTEDProducer::propagateAndUpdate(
    const TrajectoryStateOnSurface initialTSOS, const Propagator& prop, const TrackingRecHit& hit) {
  TrajectoryStateOnSurface propTSOS = prop.propagate(initialTSOS, hit.det()->surface());
  TrajectoryStateOnSurface updatedTSOS = theUpdator->update(propTSOS, hit);
  if
    UNLIKELY(!updatedTSOS.isValid()) return std::make_pair(false, updatedTSOS);
  return std::make_pair(true, updatedTSOS);
}

float SeedingOTEDProducer::computeGlobalThetaError(const VectorHit& vh, const double sigmaZ_beamSpot) {
  double derivative =
      vh.globalPosition().perp() / (pow(vh.globalPosition().z(), 2) + pow(vh.globalPosition().perp(), 2));
  double derivative2 = pow(derivative, 2);
  return pow(derivative2 * vh.lowerGlobalPosErr().czz() + derivative2 * pow(sigmaZ_beamSpot, 2), 0.5);
}

float SeedingOTEDProducer::computeInverseMomentumError(VectorHit& vh,
                                                       const float globalTheta,
                                                       const MagneticField* magField,
                                                       const double sigmaZ_beamSpot) {
  //for pT > 2GeV, 1/pT has sigma = 1/sqrt(12)
  float varianceInverseTransvMomentum = 1. / 12;
  double derivativeTheta2 = pow(cos(globalTheta) / vh.transverseMomentum(magField), 2);
  double derivativeInverseTransvMomentum2 = pow(sin(globalTheta), 2);
  float thetaError = computeGlobalThetaError(vh, sigmaZ_beamSpot);
  return pow(derivativeTheta2 * pow(thetaError, 2) + derivativeInverseTransvMomentum2 * varianceInverseTransvMomentum,
             0.5);
}

TrajectorySeed SeedingOTEDProducer::createSeed(const TrajectoryStateOnSurface& tsos,
                                               const edm::OwnVector<TrackingRecHit>& container,
                                               const DetId& id,
                                               const Propagator& prop) {
  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());
  return TrajectorySeed(seedTSOS, container, prop.propagationDirection());
}
