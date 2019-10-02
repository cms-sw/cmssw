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
  std::cout << "SeedingOT::produce() begin" << std::endl;
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
    std::cout << "BeamSpot Position: " << *(beamSpotH.product());
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

  std::cout << "SeedingOT::produce() end" << std::endl;
}

TrajectorySeedCollection SeedingOTEDProducer::run(edm::Handle<VectorHitCollectionNew> VHs) {
  TrajectorySeedCollection result;

  std::cout << "-----------------------------" << std::endl;
  printVHsOnLayer(VHs, 1);
  printVHsOnLayer(VHs, 2);
  printVHsOnLayer(VHs, 3);
  printVHsOnLayer(VHs, 4);
  printVHsOnLayer(VHs, 5);
  printVHsOnLayer(VHs, 6);
  std::cout << "-----------------------------" << std::endl;

  //check if all the first three layers have VHs
  std::vector<VectorHit> VHseedsL1 = collectVHsOnLayer(VHs, 1);
  std::vector<VectorHit> VHseedsL2 = collectVHsOnLayer(VHs, 2);
  std::vector<VectorHit> VHseedsL3 = collectVHsOnLayer(VHs, 3);
  if (VHseedsL1.empty() || VHseedsL2.empty() || VHseedsL3.empty()) {
    std::cout << "------- seeds found: " << result.size() << " ------" << std::endl;
    std::cout << "- L1 or L2 or L3 are empty! -" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    return result;
  }

  //seeds are built in the L3 of the OT
  const BarrelDetLayer* barrelOTLayer2 = measurementTracker->geometricSearchTracker()->tobLayers().at(1);
  std::cout << "VH seeds = " << VHseedsL3.size() << std::endl;

  //the search propag directiondepend on the sign of signZ*signPz, while the building is always the contrary
  Propagator* searchingPropagator = &*propagator->clone();
  Propagator* buildingPropagator = &*propagator->clone();
  buildingPropagator->setPropagationDirection(alongMomentum);

  for (auto hitL3 : VHseedsL3) {
    //building a tsos out of a VectorHit
    std::cout << "\t1a) Building a seed for the VH: " << hitL3 << std::endl;
    const TrajectoryStateOnSurface initialTSOS = buildInitialTSOS(hitL3);
    float signZ = copysign(1.0, initialTSOS.globalPosition().z());
    float signPz = copysign(1.0, initialTSOS.globalMomentum().z());

    std::cout << "\t    initialTSOS    : " << initialTSOS << std::endl;

    //set the direction of the propagator
    std::cout << "\t1b) Set the searchingPropagator direction: " << std::endl;
    if (signZ * signPz > 0.0)
      searchingPropagator->setPropagationDirection(oppositeToMomentum);
    if (signZ * signPz < 0.0)
      searchingPropagator->setPropagationDirection(alongMomentum);

    if (searchingPropagator->propagationDirection() == alongMomentum)
      std::cout << "\t    searchingPropagator along Momentum" << std::endl;
    if (searchingPropagator->propagationDirection() == oppositeToMomentum)
      std::cout << "\t    ropagator opposite To Momentum" << std::endl;

    //find vHits in layer 2
    std::cout << "-----------------------------" << std::endl;
    std::cout << "\t1c) Search/find hit in layer 2: " << std::endl;
    std::vector<TrajectoryMeasurement> measurementsL2 =
        layerMeasurements->measurements(*barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator);
    std::cout << "\t    vh compatibles on L2: " << measurementsL2.size() << std::endl;

    //other options
    //LayerMeasurements::SimpleHitContainer hits;
    //layerMeasurements->recHits(hits, *barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator);
    //std::cout << "\t    #try2  vh compatibles with recHits: " << hits.size() << std::endl;
    //auto && measurementsL2G = layerMeasurements->groupedMeasurements(*barrelOTLayer2, initialTSOS, *searchingPropagator, *estimator);
    //std::cout << "\t    #try3  vh grouped compatibles: " << measurementsL2G.size() << std::endl;

    std::vector<TrajectoryMeasurement>::iterator measurementsL2end =
        std::remove_if(measurementsL2.begin(), measurementsL2.end(), isInvalid());
    measurementsL2.erase(measurementsL2end, measurementsL2.end());
    std::cout << "\t    vh compatibles on L2(without invalidHit): " << measurementsL2.size() << std::endl;
    std::cout << "-----------------------------" << std::endl;

    if (!measurementsL2.empty()) {
      //not sure if building it everytime takes time/memory
      const DetLayer* barrelOTLayer1 = measurementTracker->geometricSearchTracker()->tobLayers().at(0);

      for (auto mL2 : measurementsL2) {
        std::cout << "\t2a) Check the searchingPropagator direction: " << std::endl;
        if (searchingPropagator->propagationDirection() == alongMomentum)
          std::cout << "\t    searchingPropagator along Momentum" << std::endl;
        if (searchingPropagator->propagationDirection() == oppositeToMomentum)
          std::cout << "\t    searchingPropagator opposite To Momentum" << std::endl;

        const TrackingRecHit* hitL2 = mL2.recHit().get();
        std::cout << "\t2b) and the VH on layer 2: " << std::endl;
        const VectorHit* vhit = dynamic_cast<const VectorHit*>(hitL2);
        std::cout << "\t    vh is valid >> " << (*vhit) << std::endl;

        //propagate to the L2 and update the TSOS
        std::cout << "\t2c) Propagation and update on L2: " << std::endl;
        std::pair<bool, TrajectoryStateOnSurface> updatedTSOS =
            propagateAndUpdate(initialTSOS, *searchingPropagator, *hitL2);
        if (!updatedTSOS.first)
          std::cout << "\t    updatedTSOS  on L2 is NOT valid  : " << updatedTSOS.second << std::endl;
        if (!updatedTSOS.first)
          continue;
        std::cout << "\t    updatedTSOS is valid  : " << updatedTSOS.second << std::endl;
        std::cout << "\t    chi2 VH/updatedTSOS  : " << estimator->estimate(updatedTSOS.second, *hitL2).second
                  << std::endl;

        //searching possible VHs in L1
        std::cout << "\t2d) Search/find hit in layer 1: " << std::endl;
        std::vector<TrajectoryMeasurement> measurementsL1 =
            layerMeasurements->measurements(*barrelOTLayer1, updatedTSOS.second, *searchingPropagator, *estimator);
        std::cout << "\t    vh compatibles on L1: " << measurementsL1.size() << std::endl;
        std::vector<TrajectoryMeasurement>::iterator measurementsL1end =
            std::remove_if(measurementsL1.begin(), measurementsL1.end(), isInvalid());
        measurementsL1.erase(measurementsL1end, measurementsL1.end());
        std::cout << "\t    vh compatibles on L1(without invalidHit): " << measurementsL1.size() << std::endl;
        std::cout << "-----------------------------" << std::endl;

        if (!measurementsL1.empty()) {
          for (auto mL1 : measurementsL1) {
            std::cout << "\t3a) Check the searchingPropagator direction: " << std::endl;
            if (searchingPropagator->propagationDirection() == alongMomentum)
              std::cout << "\t    searchingPropagator along Momentum" << std::endl;
            if (searchingPropagator->propagationDirection() == oppositeToMomentum)
              std::cout << "\t    searchingPropagator opposite To Momentum" << std::endl;

            const TrackingRecHit* hitL1 = mL1.recHit().get();
            std::cout << "\t3b) and the VH on layer 1: " << std::endl;
            const VectorHit* vhitL1 = dynamic_cast<const VectorHit*>(hitL1);
            std::cout << "\t   vh is valid >> " << (*vhitL1) << std::endl;

            //propagate to the L1 and update the TSOS
            std::cout << "\t3c) Propagation and update on L1: " << std::endl;
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL1 =
                propagateAndUpdate(updatedTSOS.second, *searchingPropagator, *hitL1);
            if (!updatedTSOSL1.first)
              std::cout << "\t    updatedTSOS  on L1 is NOT valid  : " << updatedTSOSL1.second << std::endl;
            if (!updatedTSOSL1.first)
              continue;
            std::cout << "\t    updatedTSOS  on L1   : " << updatedTSOSL1.second << std::endl;
            std::cout << "\t    chi2 VH/updatedTSOS  : " << estimator->estimate(updatedTSOSL1.second, *hitL1).second
                      << std::endl;

            std::cout << "\t3d) Creation of the Seed: " << std::endl;
            // passSelection(updatedTSOS) :
            // http://cmslxr.fnal.gov/lxr/source/FastSimulation/Muons/plugins/FastTSGFromPropagation.cc?v=CMSSW_8_1_X_2016-09-04-2300#0474
            edm::OwnVector<TrackingRecHit> container;
            container.push_back(hitL1->clone());
            container.push_back(hitL2->clone());
            container.push_back(hitL3.clone());

            //building trajectory inside-out
            std::cout << "\t3e) Building trajectory inside-out: " << std::endl;
            if (searchingPropagator->propagationDirection() == alongMomentum) {
              buildingPropagator->setPropagationDirection(oppositeToMomentum);
              std::cout << "\t    buildingPropagator opposite To Momentum" << std::endl;
            } else if (searchingPropagator->propagationDirection() == oppositeToMomentum) {
              buildingPropagator->setPropagationDirection(alongMomentum);
              std::cout << "\t    buildingPropagator along Momentum" << std::endl;
            }

            updatedTSOSL1.second.rescaleError(100);
            std::cout << "\t    updatedTSOS  on L1   : " << updatedTSOSL1.second << std::endl;

            TrajectoryStateOnSurface updatedTSOSL1_final = theUpdator->update(updatedTSOSL1.second, *hitL1);
            if
              UNLIKELY(!updatedTSOSL1_final.isValid()) continue;
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL2_final =
                propagateAndUpdate(updatedTSOSL1_final, *buildingPropagator, *hitL2);
            std::pair<bool, TrajectoryStateOnSurface> updatedTSOSL3_final =
                propagateAndUpdate(updatedTSOSL2_final.second, *buildingPropagator, hitL3);
            std::cout << "\t    updatedTSOS final on L3   : " << updatedTSOSL3_final.second << std::endl;
            TrajectorySeed ts =
                createSeed(updatedTSOSL3_final.second, container, hitL3.geographicalId(), *buildingPropagator);
            result.push_back(ts);
          }
        }

        std::cout << "-----" << std::endl;
      }
    }
  }
  std::cout << "-----------------------------" << std::endl;
  std::cout << "------- seeds found: " << result.size() << " ------" << std::endl;
  std::cout << "-----------------------------" << std::endl;

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
    //std::cout << "initial VH collection size = " << input.size() << std::endl;
    for (auto DSViter : input) {
      if (checkLayer(DSViter.id()) == layerNumber) {
        for (auto vh : DSViter) {
          VHsOnLayer.push_back(vh);
        }
      }
    }
  }

  //std::cout << "VH in layer " << layerNumber << " collection size = " << VHsOnLayer.size() << std::endl;

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
  //std::cout << "updatedTSOS  : " << updatedTSOS << std::endl;
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
  /*
  //I have already propagator and updator
  //const Propagator*  propagator = &(*propagatorHandle);
  //KFUpdator  updator;

  // Now update initial state track using information from seed hits.

  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;

  for ( unsigned int iHit = 1; iHit < container.size(); iHit++) {

    std::pair<bool, TrajectoryStateOnSurface> state;
    if(iHit==1)
      state = propagateAndUpdate(tsos, *propagator, container[iHit]);
    else
      state = propagateAndUpdate(updatedState, *propagator, container[iHit]);
    //std::cout << "-------> new state >> " << state.second << std::endl;
    if(state.first)
      updatedState = state.second;
*/
  /*
    TrajectoryStateOnSurface state = (iHit==1) ? propagator->propagate(tsos, container[iHit].det()->surface()) : propagator->propagate(updatedState, container[iHit].det()->surface());

    std::cout << "-------> new state >> " << state << std::endl;

    if (!state.isValid()) return TrajectorySeed();

    //SeedingHitSet::ConstRecHitPointer   tth = hits[iHit]; 
    //std::unique_ptr<BaseTrackerRecHit> newtth(refitHit( tth, state));
    //if (!checkHit(state,&*newtth)) return;
  
    std::cout << "-------> updated state >> " << state << std::endl;
    updatedState =  theUpdator->update(state, container[iHit]);
    if (!updatedState.isValid()) return TrajectorySeed();

    //seedHits.push_back(newtth.release());
*/
  //  }

  //if(!hit) return;

  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());
  return TrajectorySeed(seedTSOS, container, prop.propagationDirection());
  //if ( !filter || filter->compatible(seed)) seedCollection.push_back(std::move(seed));
}
