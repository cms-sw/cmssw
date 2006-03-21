#include "RecoTracker/TrackProducer/interface/TrackProducer.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"

// constructors and destructor
TrackProducer::TrackProducer(const edm::ParameterSet& iConfig):
  theAlgo(iConfig), conf_(iConfig)
{
   //register your products
   produces<reco::TrackCollection>();
   produces<reco::TrackExtraCollection>();
}

TrackProducer::~TrackProducer(){ }

// member functions
// ------------ method called to produce the data  ------------
void TrackProducer::produce(edm::Event& theEvent, const edm::EventSetup& setup)
{
  // create empty output collections
  std::auto_ptr<reco::TrackCollection> outputTColl;
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl;

  //get geometry
  edm::ESHandle<TrackerGeometry> theG;
  setup.get<TrackerDigiGeometryRecord>().get(theG);

  //get magnetic field
  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);
  
  //
  // temporary!!!
  //

  //construct Propagator, Updator and Estimator
  thePropagator = new  AnalyticalPropagator( theMF.product(), alongMomentum);
  theUpdator = new KFUpdator();
  theEstimator = new Chi2MeasurementEstimator(1., 3.);//parameters should come from parameter set
  
  //build the fitter
  const KFTrajectoryFitter * theFitter = new KFTrajectoryFitter(thePropagator,theUpdator,theEstimator);

  //run the algorithm  
  theAlgo.run(theG.product(), theMF.product(), theTCCollection, theFitter, outputTColl, outputTEColl);

  //put the TrackCollection and TrackExtraCollection in the event
  theEvent.put(outputTColl);
  theEvent.put(outputTEColl);

  //
  // temporary!
  //

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackProducer)
