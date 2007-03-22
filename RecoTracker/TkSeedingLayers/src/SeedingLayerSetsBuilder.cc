#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"


#include <iostream>
#include <sstream>
#include <ostream>
#include <fstream>


using namespace ctfseeding;
using namespace std;

typedef std::vector<std::string>::const_iterator IS;
typedef std::vector<std::vector<std::string> >::const_iterator IT;


SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg)
 : theNumberOfLayersInSet(cfg.getParameter<int>("layerSetSize")) 
{
  std::vector<std::string> names = cfg.getParameter<std::vector<std::string> >("layerList");
  init(names);
}


SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(
    unsigned int NumLayers, 
    const std::vector<std::string> & layerNames)
  : theNumberOfLayersInSet(NumLayers)
{
  init(layerNames);
}


void SeedingLayerSetsBuilder::init( const std::vector<std::string> & layerNames)
{
  for (std::vector<std::string>::const_iterator is=layerNames.begin(); is < layerNames.end(); ++is) {
    vector<std::string> layersInSet;
    string line = *is;
    string::size_type pos=0;
    while (pos != string::npos ) {
      pos=line.find("+");
      string layer = line.substr(0,pos);
      layersInSet.push_back(layer);
      line=line.substr(pos+1,string::npos); 
    }
    theLayersInSetNames.push_back(layersInSet);
  }
  std::ostringstream str;
  for (IT it = theLayersInSetNames.begin(); it != theLayersInSetNames.end(); it++) {
    str << "SET: ";
    for (IS is = it->begin(); is != it->end(); is++)  str << *is <<" ";  
    str << std::endl;
  }
  std::cout << str.str() << std::endl;
}

SeedingLayerSets SeedingLayerSetsBuilder::layers(const edm::EventSetup& es) const
{
  typedef std::vector<SeedingLayer> Set;
  SeedingLayerSets  result;

  edm::ESHandle<GeometricSearchTracker> tracker;
  es.get<TrackerRecoGeometryRecord>().get( tracker );

  std::vector<BarrelDetLayer*>  pbl  = tracker->barrelLayers();
  std::vector<ForwardDetLayer*> fpos=tracker->posForwardLayers();
  std::vector<ForwardDetLayer*> fneg=tracker->negForwardLayers();
  std::vector<BarrelDetLayer*>  tib = tracker->tibLayers();
  
  for (IT it = theLayersInSetNames.begin(); it != theLayersInSetNames.end(); it++) {
    Set set;
    bool setOK = true;
    for (IS is = it->begin(), isEnd = it->end(); is < isEnd; ++is) {

      std::string name = (*is);
      const DetLayer * detLayer =0;
      SeedingLayer::Side side=SeedingLayer::Barrel;
      int idLayer = 0;
      bool nameOK = true;

      if      (name=="BPix1")  { idLayer=1; side=SeedingLayer::Barrel; detLayer=pbl[idLayer-1]; }
      else if (name=="BPix2")  { idLayer=2; side=SeedingLayer::Barrel; detLayer=pbl[idLayer-1]; }
      else if (name=="BPix3")  { idLayer=3; side=SeedingLayer::Barrel; detLayer=pbl[idLayer-1]; }
      else if (name=="FPix1_pos")  {idLayer=1; detLayer = fpos[idLayer-1]; side = SeedingLayer::PosEndcap; }
      else if (name=="FPix2_pos")  {idLayer=2; detLayer = fpos[idLayer-1]; side = SeedingLayer::PosEndcap; }
      else if (name=="FPix1_neg")  {idLayer=1; detLayer = fneg[idLayer-1]; side = SeedingLayer::NegEndcap; }
      else if (name=="FPix2_neg")  {idLayer=2; detLayer = fneg[idLayer-1]; side = SeedingLayer::NegEndcap; }

      else if (name=="TIB1")  { idLayer=1; side = SeedingLayer::Barrel;  detLayer = tib[idLayer-1]; }

      else {
        nameOK = false;
        setOK = false;
      }

      

      if(nameOK) set.push_back( SeedingLayer(detLayer, name, side,idLayer) );
    
    }
    if(setOK) result.push_back(set);
  }
  return result;
}
