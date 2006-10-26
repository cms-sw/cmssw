//
// Package:         RecoTracker/TkSeedGeneratorFromMuon
// Class:           RegionalPixelSeedGeneratorFromMuon
// 

#include <iostream>
#include <memory>
#include <string>

#include "RecoTracker/TkSeedGenerator/interface/RegionalPixelSeedGeneratorFromMuon.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "DataFormats/Math/interface/Vector3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

RegionalPixelSeedGeneratorFromMuon::RegionalPixelSeedGeneratorFromMuon(edm::ParameterSet const& conf) : 
  conf_(conf),combinatorialSeedGenerator(conf)
{
  edm::LogInfo ("RegionalPixelSeedGeneratorFromMuon")<<"Enter the RegionalPixelSeedGeneratorFromMuon";
  produces<TrajectorySeedCollection>();

  ptmin=conf_.getParameter<double>("ptMin");
  originradius=conf_.getParameter<double>("originRadius");
  halflength=conf_.getParameter<double>("originHalfLength");
  vertexSrc=conf_.getParameter<string>("vertexSrc");
  deltaEta = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi = conf_.getParameter<double>("deltaPhiRegion");
  jetSrc = conf_.getParameter<edm::InputTag>("MuonSrc");

}

// Virtual destructor needed.
RegionalPixelSeedGeneratorFromMuon::~RegionalPixelSeedGeneratorFromMuon() { }  

// Functions that gets called by framework every event
void RegionalPixelSeedGeneratorFromMuon::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  edm::Handle<reco::VertexCollection> vertices;
  e.getByLabel(vertexSrc,vertices);
  const reco::VertexCollection vertCollection = *(vertices.product());
  reco::VertexCollection::const_iterator ci = vertCollection.begin();
  if(vertCollection.size() == 0) return;
  originz = ci->z();



  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  e.getByLabel(hitProducer, pixelHits);
  //  e.getByType(pixelHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());    

  //
  
  //Get the jet direction
  edm::Handle<MuonCollection> jets;
  e.getByLabel(jetSrc, jets);

  GlobalVector globalVector(0,0,1);

  //    if(jets->size() > 0)
  //    {
     MuonCollection::const_iterator iJet = jets->begin();
      for(;iJet != jets->end();iJet++)
	{
	  
	  GlobalVector jetVector((iJet)->p4().x(),(iJet)->p4().y(),(iJet)->p4().z());
	  globalVector = jetVector;
	  
	  
	  RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(globalVector,
											       GlobalPoint(0,0,originz), 
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
