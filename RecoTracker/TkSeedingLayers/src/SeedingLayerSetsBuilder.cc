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

namespace {
  std::tuple<GeomDetEnumerators::SubDetector,
             SeedingLayer::Side,
             int> nameToEnumId(const std::string& name) {
    GeomDetEnumerators::SubDetector subdet = GeomDetEnumerators::invalidDet;
    SeedingLayer::Side side = SeedingLayer::Barrel;
    int idLayer = 0;

    size_t index;
    //
    // BPIX
    //
    if ((index = name.find("BPix")) != string::npos) {
      subdet = GeomDetEnumerators::PixelBarrel;
      side = SeedingLayer::Barrel;
      idLayer = atoi(name.substr(index+4,1).c_str());
    }
    //
    // FPIX
    //
    else if ((index = name.find("FPix")) != string::npos) {
      subdet = GeomDetEnumerators::PixelEndcap;
      idLayer = atoi(name.substr(index+4,1).c_str());
      if ( name.find("pos") != string::npos ) {
        side = SeedingLayer::PosEndcap;
      } else {
        side = SeedingLayer::NegEndcap;
      }
    }
    //
    // TIB
    //
    else if ((index = name.find("TIB")) != string::npos) {
      subdet = GeomDetEnumerators::TIB;
      side = SeedingLayer::Barrel;
      idLayer = atoi(name.substr(index+3,1).c_str());
    }
    //
    // TID
    //
    else if ((index = name.find("TID")) != string::npos) {
      subdet = GeomDetEnumerators::TID;
      idLayer = atoi(name.substr(index+3,1).c_str());
      if ( name.find("pos") != string::npos ) {
        side = SeedingLayer::PosEndcap;
      } else {
        side = SeedingLayer::NegEndcap;
      }
    }
    //
    // TOB
    //
    else if ((index = name.find("TOB")) != string::npos) {
      subdet = GeomDetEnumerators::TOB;
      side = SeedingLayer::Barrel;
      idLayer = atoi(name.substr(index+3,1).c_str());
    }
    //
    // TEC
    //
    else if ((index = name.find("TEC")) != string::npos) {
      subdet = GeomDetEnumerators::TEC;
      idLayer = atoi(name.substr(index+3,1).c_str());
      if ( name.find("pos") != string::npos ) {
        side = SeedingLayer::PosEndcap;
      } else {
        side = SeedingLayer::NegEndcap;
      }
    }
    return std::make_tuple(subdet, side, idLayer);
  }
}

SeedingLayerSetsBuilder::LayerSpec::LayerSpec() {}
SeedingLayerSetsBuilder::LayerSpec::~LayerSpec() {}

std::string SeedingLayerSetsBuilder::LayerSpec::print() const
{
  std::ostringstream str;
  str << "Layer="<<name<<", hitBldr: "<<hitBuilder;

  str << ", useRingSelector: ";
  HitExtractorSTRP *ext = nullptr;
  if((ext = dynamic_cast<HitExtractorSTRP *>(extractor.get())) &&
     ext->useRingSelector()) {
    auto minMaxRing = ext->getMinMaxRing();
    str <<"true,"<<" Rings: ("<< std::get<0>(minMaxRing) <<","<< std::get<1>(minMaxRing) <<")"; 
  } else  str<<"false";

  return str.str();
}

SeedingLayerSetsBuilder::SeedingLayerSetsBuilder() {}
SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector&& iC):
  SeedingLayerSetsBuilder(cfg, iC)
{}
SeedingLayerSetsBuilder::SeedingLayerSetsBuilder(const edm::ParameterSet & cfg, edm::ConsumesCollector& iC)
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

  map<string, std::pair<std::size_t, std::size_t> > mapConfig; // for debug printout only!

  int layerSetId = 0;
  for (IT it = layerNamesInSets.begin(); it != layerNamesInSets.end(); it++) {
    vector<LayerSpec> layersInSet;
    for (IS is = it->begin(); is != it->end(); is++) {
      LayerSpec layer;

      layer.name = *is;
      //std::cout << "layer name in config: " << *is << std::endl;
      edm::ParameterSet cfgLayer = layerConfig(layer.name, cfg);

      layer.usePixelHitProducer = true;
      if (cfgLayer.exists("HitProducer")) {
          layer.pixelHitProducer = cfgLayer.getParameter<string>("HitProducer"); 
      }else{
          layer.usePixelHitProducer = false;
      }
      bool skipClusters = cfgLayer.exists("skipClusters");
      if (skipClusters) {
        LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" ready for skipping";
      }
      else{
        LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" not skipping ";
      }
      layer.hitBuilder  = cfgLayer.getParameter<string>("TTRHBuilder");

      auto subdetData = nameToEnumId(layer.name);
      layer.subdet = std::get<0>(subdetData);
      layer.side = std::get<1>(subdetData);
      layer.idLayer = std::get<2>(subdetData);
      if(layer.subdet == GeomDetEnumerators::PixelBarrel ||
         layer.subdet == GeomDetEnumerators::PixelEndcap) {
        layer.extractor = std::make_shared<HitExtractorPIX>(layer.side, layer.idLayer, layer.pixelHitProducer, iC);
      }
      else if(layer.subdet != GeomDetEnumerators::invalidDet) {
        std::shared_ptr<HitExtractorSTRP> extractor = std::make_shared<HitExtractorSTRP>(layer.side, layer.idLayer, iC);
        if (cfgLayer.exists("matchedRecHits")) {
          extractor->useMatchedHits(cfgLayer.getParameter<edm::InputTag>("matchedRecHits"), iC);
        }
        if (cfgLayer.exists("rphiRecHits")) {
	  extractor->useRPhiHits(cfgLayer.getParameter<edm::InputTag>("rphiRecHits"), iC);
        }
        if (cfgLayer.exists("stereoRecHits")) {
          extractor->useStereoHits(cfgLayer.getParameter<edm::InputTag>("stereoRecHits"), iC);
        }
        if (cfgLayer.exists("useRingSlector") && cfgLayer.getParameter<bool>("useRingSlector")) {
          extractor->useRingSelector(cfgLayer.getParameter<int>("minRing"),
                                     cfgLayer.getParameter<int>("maxRing"));
        }
        bool useSimpleRphiHitsCleaner = cfgLayer.exists("useSimpleRphiHitsCleaner") ? cfgLayer.getParameter<bool>("useSimpleRphiHitsCleaner") : true;
        extractor->useSimpleRphiHitsCleaner(useSimpleRphiHitsCleaner);

        double minAbsZ = cfgLayer.exists("MinAbsZ") ? cfgLayer.getParameter<double>("MinAbsZ") : 0.;
        if(minAbsZ > 0.) {
          extractor->setMinAbsZ(minAbsZ);
        }
        if(skipClusters) {
          bool useProjection = cfgLayer.exists("useProjection") ? cfgLayer.getParameter<bool>("useProjection") : false;
          if(useProjection) {
            LogDebug("SeedingLayerSetsBuilder")<<layer.name<<" will project partially masked matched rechit";
          }
          else {
	    extractor->setNoProjection();
          }
        }

        layer.extractor = extractor;
      }
      if(layer.extractor && skipClusters) {
        layer.extractor->useSkipClusters(cfgLayer.getParameter<edm::InputTag>("skipClusters"), iC);
      }


      mapConfig[layer.name]=std::make_pair(theLayersInSets.size(), layersInSet.size());
      if (nameToId.find(layer.name)==nameToId.end()) {
	std::string name = layer.name;
	nameToId.insert( std::pair<std::string,int>(name,layerSetId) );
	layerSetId++;	
      }
      layersInSet.push_back(layer);
    }
    theLayersInSets.push_back(layersInSet);
  }

  // debug printout
  // The following should not be set to cout
//  for(const auto& nameIndices: mapConfig) {
//    std::cout << theLayersInSets[nameIndices.second.first][nameIndices.second.second].print() << std::endl;
//  }
}

SeedingLayerSetsBuilder::~SeedingLayerSetsBuilder() {}

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
      const DetLayer * detLayer =0;
      bool nameOK = true;
      int index = layer.idLayer-1;
      
      if (layer.subdet == GeomDetEnumerators::PixelBarrel) {
        detLayer = bpx[index]; 
      }
      else if (layer.subdet == GeomDetEnumerators::PixelEndcap) {
        if (layer.side == SeedingLayer::PosEndcap) {
          detLayer = fpx_pos[index];
        } else {
          detLayer = fpx_neg[index];
        }
      }
      else if (layer.subdet == GeomDetEnumerators::TIB) {
        detLayer = tib[index];
      }
      else if (layer.subdet == GeomDetEnumerators::TID) {
        if (layer.side == SeedingLayer::PosEndcap) {
          detLayer = tid_pos[index];
        } else {
          detLayer = tid_neg[index];
        }
      }
      else if (layer.subdet == GeomDetEnumerators::TOB) {
        detLayer = tob[index];
      }
      else if (layer.subdet == GeomDetEnumerators::TEC) {
        if (layer.side == SeedingLayer::PosEndcap) {
          detLayer = tec_pos[index];
        } else {
          detLayer = tec_neg[index];
        }
      }
      else {
        nameOK = false;
        setOK = false;
      }

      if(nameOK) {
        std::unique_ptr<HitExtractor> extractor(layer.extractor->clone());
        if ( detLayer->subDetector() != GeomDetEnumerators::PixelBarrel &&
             detLayer->subDetector() != GeomDetEnumerators::PixelEndcap) {
          dynamic_cast<HitExtractorSTRP *>(extractor.get())->setDetLayer(detLayer);
        }

        edm::ESHandle<TransientTrackingRecHitBuilder> builder;
        es.get<TransientRecHitRecord>().get(layer.hitBuilder, builder);

	auto it = nameToId.find(layer.name);
	if (it==nameToId.end()) {
	  edm::LogError("SeedingLayerSetsBuilder")<<"nameToId map mismatch! Could not find: "<<layer.name;
	  return result;
	}
	int layerSetId = it->second;
        set.push_back( SeedingLayer( layer.name, layerSetId, detLayer, builder.product(), extractor.release()));
      }
    
    }
    if(setOK) result.push_back(set);
  }
  return result;
}

bool SeedingLayerSetsBuilder::check(const edm::EventSetup& es) {
  // We want to evaluate both in the first invocation (to properly
  // initialize ESWatcher), and this way we avoid one branch compared
  // to || (should be tiny effect)
  return geometryWatcher_.check(es) | trhWatcher_.check(es);
}
