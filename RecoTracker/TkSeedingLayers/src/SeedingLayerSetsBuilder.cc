#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "HitExtractorPIX.h"
#include "HitExtractorSTRP.h"

#include <iostream>
#include <sstream>
#include <ostream>
#include <fstream>
#include <map>


using namespace ctfseeding;
using namespace std;


std::string SeedingLayerSetsBuilder::LayerSpec::print() const
{
  std::ostringstream str;
  str << "Layer="<<name<<", hitBldr: "<<hitBuilder<<", useErrorsFromParam: ";
  if (useErrorsFromParam) {
     str <<"true,"<<" errRPhi: "<<hitErrorRPhi<<", errRZ: "<<hitErrorRZ; 
  }
  else str<<"false";

  str << ", useRingSelector: ";
  if (useRingSelector) {
    str <<"true,"<<" Rings: ("<<minRing<<","<<maxRing<<")"; 
  } else  str<<"false";

  return str.str();
}

SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg)
{
  std::vector<std::string> namesPset = cfg.getParameter<std::vector<std::string> >("layerList");
  std::vector<std::vector<std::string> > layerNamesInSets = this->layerNamesInSets(namesPset);

  // debug printout of layers
  typedef std::vector<std::string>::const_iterator IS;
  typedef std::vector<std::vector<std::string> >::const_iterator IT;
  std::ostringstream str;
  // The following should not be set to cout
//  for (IT it = layerNamesInSets.begin(); it != layerNamesInSets.end(); it++) {
//    str << "SET: ";
//    for (IS is = it->begin(); is != it->end(); is++)  str << *is <<" ";  
//    str << std::endl;
//  }
//  std::cout << str.str() << std::endl;

  map<string,LayerSpec> mapConfig; // for debug printout only!

  for (IT it = layerNamesInSets.begin(); it != layerNamesInSets.end(); it++) {
    vector<LayerSpec> layersInSet;
    for (IS is = it->begin(); is != it->end(); is++) {
      LayerSpec layer;

      layer.name = *is;
      //std::cout << "layer name in config: " << *is << std::endl;
      edm::ParameterSet cfgLayer = layerConfig(layer.name, cfg);

      layer.usePixelHitProducer = true;
      layer.useMatchedRecHits = true; 
      layer.useRPhiRecHits = true;
      layer.useStereoRecHits = true;
      if (cfgLayer.exists("HitProducer")) {
          layer.pixelHitProducer = cfgLayer.getParameter<string>("HitProducer"); 
      }else{
          layer.usePixelHitProducer = false;
      }
      if (cfgLayer.exists("matchedRecHits")) {
          layer.matchedRecHits = cfgLayer.getParameter<edm::InputTag>("matchedRecHits"); 
      }else{
          layer.useMatchedRecHits = false;
      }
      if (cfgLayer.exists("rphiRecHits")) {
          layer.rphiRecHits = cfgLayer.getParameter<edm::InputTag>("rphiRecHits"); 
      }else{
          layer.useRPhiRecHits = false;
      }
      if (cfgLayer.exists("stereoRecHits")) {
          layer.stereoRecHits= cfgLayer.getParameter<edm::InputTag>("stereoRecHits"); 
      }else{
          layer.useStereoRecHits = false;
      }
      if (cfgLayer.exists("skipClusters")){
	LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" ready for skipping (1)";
	layer.clustersToSkip = cfgLayer.getParameter<edm::InputTag>("skipClusters");
	layer.skipClusters=true;
      }else{
	layer.skipClusters=false;
      }
      if (layer.skipClusters){
	if (cfgLayer.exists("useProjection")){
	  LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" will project partially masked matched rechit";
	  layer.useProjection=cfgLayer.getParameter<bool>("useProjection");
	}
	else{
	  layer.useProjection=false;
	}
      }
      layer.hitBuilder  = cfgLayer.getParameter<string>("TTRHBuilder");

      layer.useErrorsFromParam = cfgLayer.exists("useErrorsFromParam") ? cfgLayer.getParameter<bool>("useErrorsFromParam") : false; 
      if(layer.useErrorsFromParam) {
        layer.hitErrorRPhi = cfgLayer.getParameter<double>("hitErrorRPhi");
        layer.hitErrorRZ   = cfgLayer.getParameter<double>("hitErrorRZ");
      }

      layer.useRingSelector = cfgLayer.exists("useRingSlector") ? cfgLayer.getParameter<bool>("useRingSlector") : false;
      if (layer.useRingSelector) {
        layer.minRing = cfgLayer.getParameter<int>("minRing");
        layer.maxRing = cfgLayer.getParameter<int>("maxRing");
      }

      layer.useSimpleRphiHitsCleaner = cfgLayer.exists("useSimpleRphiHitsCleaner") ? cfgLayer.getParameter<bool>("useSimpleRphiHitsCleaner") : true; 

      layersInSet.push_back(layer);
      mapConfig[layer.name]=layer;
    }
    theLayersInSets.push_back(layersInSet);
  }

  // debug printout
  // The following should not be set to cout
//  for (map<string,LayerSpec>::const_iterator im = mapConfig.begin(); im != mapConfig.end(); im++) {
//    std::cout << (*im).second.print() << std::endl; 
//  }
}

edm::ParameterSet SeedingLayerSetsBuilder::layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const
{
  edm::ParameterSet result;
   
  for (string::size_type iEnd=nameLayer.size(); iEnd > 0; --iEnd) {
    string name = nameLayer.substr(0,iEnd);
    if (cfg.exists(name)) return cfg.getParameter<edm::ParameterSet>(name);
  } 
  cout <<"configuration for layer: "<<nameLayer<<" not found, job will probably crash!"<<endl;
  return result;
}

vector<vector<string> > SeedingLayerSetsBuilder::layerNamesInSets( const vector<string> & namesPSet)
{
  std::vector<std::vector<std::string> > result; 
  for (std::vector<std::string>::const_iterator is=namesPSet.begin(); is < namesPSet.end(); ++is) {
    vector<std::string> layersInSet;
    string line = *is;
    string::size_type pos=0;
    while (pos != string::npos ) {
      pos=line.find("+");
      string layer = line.substr(0,pos);
      layersInSet.push_back(layer);
      line=line.substr(pos+1,string::npos); 
    }
    result.push_back(layersInSet);
  }
  return result;
}

SeedingLayerSets SeedingLayerSetsBuilder::layers(const edm::EventSetup& es) const
{
  typedef std::vector<SeedingLayer> Set;
  SeedingLayerSets  result;

  edm::ESHandle<GeometricSearchTracker> tracker;
  es.get<TrackerRecoGeometryRecord>().get( tracker );

  std::vector<BarrelDetLayer*>  bpx  = tracker->barrelLayers();
  std::vector<BarrelDetLayer*>  tib  = tracker->tibLayers();
  std::vector<BarrelDetLayer*>  tob  = tracker->tobLayers();

  std::vector<ForwardDetLayer*> fpx_pos = tracker->posForwardLayers();
  std::vector<ForwardDetLayer*> tid_pos = tracker->posTidLayers();
  std::vector<ForwardDetLayer*> tec_pos = tracker->posTecLayers();

  std::vector<ForwardDetLayer*> fpx_neg = tracker->negForwardLayers();
  std::vector<ForwardDetLayer*> tid_neg = tracker->negTidLayers();
  std::vector<ForwardDetLayer*> tec_neg = tracker->negTecLayers();
  
  typedef std::vector<std::vector<LayerSpec> >::const_iterator IT;
  typedef std::vector<LayerSpec>::const_iterator IS;

  for (IT it = theLayersInSets.begin(); it != theLayersInSets.end(); it++) {
    Set set;
    bool setOK = true;
    for (IS is = it->begin(), isEnd = it->end(); is < isEnd; ++is) {
      const LayerSpec & layer = (*is);
      std::string name = layer.name;
      const DetLayer * detLayer =0;
      SeedingLayer::Side side=SeedingLayer::Barrel;
      int idLayer = 0;
      bool nameOK = true;
      HitExtractor * extractor = 0; 
      
      //
      // BPIX
      //
      if (name.substr(0,4) == "BPix") {
        idLayer = atoi(name.substr(4,1).c_str());
        side=SeedingLayer::Barrel;
        detLayer=bpx[idLayer-1]; 
      }
      //
      // FPIX
      //
      else if (name.substr(0,4) == "FPix") {
        idLayer = atoi(name.substr(4,1).c_str());
        if ( name.find("pos") != string::npos ) {
          side = SeedingLayer::PosEndcap;
          detLayer = fpx_pos[idLayer-1];
        } else {
          side = SeedingLayer::NegEndcap;
          detLayer = fpx_neg[idLayer-1];
        }
      }
      //
      // TIB
      //
      else if (name.substr(0,3) == "TIB") {
        idLayer = atoi(name.substr(3,1).c_str());
        side=SeedingLayer::Barrel;
        detLayer=tib[idLayer-1];
      }
      //
      // TID
      //
      else if (name.substr(0,3) == "TID") {
        idLayer = atoi(name.substr(3,1).c_str());
        if ( name.find("pos") != string::npos ) {
          side = SeedingLayer::PosEndcap;
          detLayer = tid_pos[idLayer-1];
        } else {
          side = SeedingLayer::NegEndcap;
          detLayer = tid_neg[idLayer-1];
        }
      }
      //
      // TOB
      //
      else if (name.substr(0,3) == "TOB") {
        idLayer = atoi(name.substr(3,1).c_str());
        side=SeedingLayer::Barrel;
        detLayer=tob[idLayer-1];
      }
      //
      // TEC
      //
      else if (name.substr(0,3) == "TEC") {
        idLayer = atoi(name.substr(3,1).c_str());
        if ( name.find("pos") != string::npos ) {
          side = SeedingLayer::PosEndcap;
          detLayer = tec_pos[idLayer-1];
        } else {
          side = SeedingLayer::NegEndcap;
          detLayer = tec_neg[idLayer-1];
        }
      }
      else {
        nameOK = false;
        setOK = false;
      }

      if(nameOK) {
        if ( detLayer->subDetector() == GeomDetEnumerators::PixelBarrel ||
             detLayer->subDetector() == GeomDetEnumerators::PixelEndcap) {
          extractor = new HitExtractorPIX(side,idLayer,layer.pixelHitProducer);
        } else {
          HitExtractorSTRP extSTRP(detLayer,side,idLayer);
          if (layer.useMatchedRecHits) extSTRP.useMatchedHits(layer.matchedRecHits);
          if (layer.useRPhiRecHits)    extSTRP.useRPhiHits(layer.rphiRecHits);
          if (layer.useStereoRecHits)  extSTRP.useStereoHits(layer.stereoRecHits);
          if (layer.useRingSelector)   extSTRP.useRingSelector(layer.minRing,layer.maxRing);
	  extSTRP.useSimpleRphiHitsCleaner(layer.useSimpleRphiHitsCleaner);
	  if (layer.skipClusters && !layer.useProjection)
	    extSTRP.setNoProjection();
          extractor = extSTRP.clone();
        }
	if (layer.skipClusters) {
	  LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" ready for skipping (2)";
	  extractor->useSkipClusters(layer.clustersToSkip);
	}
	else{
	  LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" not skipping ";
	}

        edm::ESHandle<TransientTrackingRecHitBuilder> builder;
        es.get<TransientRecHitRecord>().get(layer.hitBuilder, builder);

        if (layer.useErrorsFromParam) {
          set.push_back( SeedingLayer( name, detLayer, builder.product(), 
                                       extractor, true, layer.hitErrorRPhi,layer.hitErrorRZ));
        } else {
          set.push_back( SeedingLayer( name, detLayer, builder.product(), extractor));
        }
      }
    
    }
    if(setOK) result.push_back(set);
  }
  return result;
}
