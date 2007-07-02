//
// Package:         RecoTracker/TkSeedGenerator
// Class:           RegionalPixelSeedGeneratorFromCandidate
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromCandidate.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "DataFormats/Math/interface/Vector3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

RegionalPixelSeedGeneratorFromCandidate::RegionalPixelSeedGeneratorFromCandidate(edm::ParameterSet const& conf) : 
  conf_(conf),combinatorialSeedGenerator(conf)
{
  edm::LogInfo ("RegionalPixelSeedGeneratorFromCandidate")<<"Enter the RegionalPixelSeedGeneratorFromCandidate";
  produces<TrajectorySeedCollection>();

  ptmin=conf_.getParameter<double>("ptMin");
  originradius=conf_.getParameter<double>("originRadius");
  halflength=conf_.getParameter<double>("originHalfLength");
//   vertexSrc=conf_.getParameter<string>("vertexSrc");
  deltaEta = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi = conf_.getParameter<double>("deltaPhiRegion");
  candSrc = conf_.getParameter<edm::InputTag>("CandSrc");

}

// Virtual destructor needed.
RegionalPixelSeedGeneratorFromCandidate::~RegionalPixelSeedGeneratorFromCandidate() { }  

// Functions that gets called by framework every event
void RegionalPixelSeedGeneratorFromCandidate::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
//   edm::Handle<reco::VertexCollection> vertices;
//   e.getByLabel(vertexSrc,vertices);
//   const reco::VertexCollection vertCollection = *(vertices.product());
//   reco::VertexCollection::const_iterator ci = vertCollection.begin();
//   if(vertCollection.size() == 0) return;
//   originz = ci->z();



  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, pixelHits);
  //  e.getByType(pixelHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());    

  //
  
  //Get the jet direction
  edm::Handle<CandidateCollection> candidates;
  e.getByLabel(candSrc, candidates);

  GlobalVector globalVector(0,0,1);

  //    if(jets->size() > 0)
  //    {
      CandidateCollection::const_iterator iCandidate = candidates->begin();
      for(;iCandidate != candidates->end();iCandidate++)
	{
	  
	  GlobalVector candidateVector((iCandidate)->p4().x(),(iCandidate)->p4().y(),(iCandidate)->p4().z());
	  globalVector = candidateVector;
	  GlobalPoint globalPoint((iCandidate)->vertex().x(),(iCandidate)->vertex().y(),(iCandidate)->vertex().z());
	  
	  RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(globalVector,
// 											       GlobalPoint(0,0,originz), 
													globalPoint,
											       ptmin,
											       originradius,
											       halflength,
											       deltaEta,
											       deltaPhi);
	  
	  combinatorialSeedGenerator.init(*pixelHits,es);
	  combinatorialSeedGenerator.run(*etaphiRegion,*output,es);
	  // write output to file
	}
      //    }   

    LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();
    e.put(output);
}
