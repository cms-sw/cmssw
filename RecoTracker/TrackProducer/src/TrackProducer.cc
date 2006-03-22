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
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

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
  // get the fitter from the ES
  //
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  edm::ESHandle<TrajectoryFitter> theFitter;
  setup.get<TrackingComponentsRecord>().get(fitterName,theFitter);
  //
  // get also the propagator
  //
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
  edm::ESHandle<Propagator> thePropagator;
  setup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  //run the algorithm  
  theAlgo.run(theG.product(), theMF.product(), theTCCollection, theFitter.product(), thePropagator.product(), outputTColl, outputTEColl);

  //put the TrackCollection and TrackExtraCollection in the event
  theEvent.put(outputTColl);
  theEvent.put(outputTEColl);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackProducer)
