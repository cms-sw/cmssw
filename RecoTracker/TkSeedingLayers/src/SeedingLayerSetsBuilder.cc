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

#include "RecoLocalTracker/SiStripClusterizer/interface/ClusterChargeCut.h"

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

SeedingLayerSetsBuilder::LayerSpec::LayerSpec(unsigned short index, const std::string& layerName, const edm::ParameterSet& cfgLayer, edm::ConsumesCollector& iC):
  nameIndex(index),
  hitBuilder(cfgLayer.getParameter<string>("TTRHBuilder"))
{
  usePixelHitProducer = false;
  if (cfgLayer.exists("HitProducer")) {
    pixelHitProducer = cfgLayer.getParameter<string>("HitProducer");
    usePixelHitProducer = true;
  }

  bool skipClusters = cfgLayer.exists("skipClusters");
  if (skipClusters) {
    LogDebug("SeedingLayerSetsBuilder")<<layerName<<" ready for skipping";
  }
  else{
    LogDebug("SeedingLayerSetsBuilder")<<layerName<<" not skipping ";
  }

  auto subdetData = nameToEnumId(layerName);
  subdet = std::get<0>(subdetData);
  side = std::get<1>(subdetData);
  idLayer = std::get<2>(subdetData);
  if(subdet == GeomDetEnumerators::PixelBarrel ||
     subdet == GeomDetEnumerators::PixelEndcap) {
    extractor = std::make_shared<HitExtractorPIX>(side, idLayer, pixelHitProducer, iC);
  }
  else if(subdet != GeomDetEnumerators::invalidDet) { // strip
    std::shared_ptr<HitExtractorSTRP> extr = std::make_shared<HitExtractorSTRP>(subdet, side, idLayer, clusterChargeCut(cfgLayer) );
    if (cfgLayer.exists("matchedRecHits")) {
      extr->useMatchedHits(cfgLayer.getParameter<edm::InputTag>("matchedRecHits"), iC);
    }
    if (cfgLayer.exists("rphiRecHits")) {
      extr->useRPhiHits(cfgLayer.getParameter<edm::InputTag>("rphiRecHits"), iC);
    }
    if (cfgLayer.exists("stereoRecHits")) {
      extr->useStereoHits(cfgLayer.getParameter<edm::InputTag>("stereoRecHits"), iC);
    }
    if (cfgLayer.exists("useRingSlector") && cfgLayer.getParameter<bool>("useRingSlector")) {
      extr->useRingSelector(cfgLayer.getParameter<int>("minRing"),
                                 cfgLayer.getParameter<int>("maxRing"));
    }
    bool useSimpleRphiHitsCleaner = cfgLayer.exists("useSimpleRphiHitsCleaner") ? cfgLayer.getParameter<bool>("useSimpleRphiHitsCleaner") : true;
    extr->useSimpleRphiHitsCleaner(useSimpleRphiHitsCleaner);

    double minAbsZ = cfgLayer.exists("MinAbsZ") ? cfgLayer.getParameter<double>("MinAbsZ") : 0.;
    if(minAbsZ > 0.) {
      extr->setMinAbsZ(minAbsZ);
    }
    if(skipClusters) {
      bool useProjection = cfgLayer.exists("useProjection") ? cfgLayer.getParameter<bool>("useProjection") : false;
      if(useProjection) {
        LogDebug("SeedingLayerSetsBuilder")<<layerName<<" will project partially masked matched rechit";
      }
      else {
        extr->setNoProjection();
      }
    }
    extractor = std::move(extr);
  }
  if(extractor && skipClusters) {
    extractor->useSkipClusters(cfgLayer.getParameter<edm::InputTag>("skipClusters"), iC);
  }
}
SeedingLayerSetsBuilder::LayerSpec::~LayerSpec() {}

std::string SeedingLayerSetsBuilder::LayerSpec::print(const std::vector<std::string>& names) const
{
  std::ostringstream str;
  str << "Layer="<<names[nameIndex]<<", hitBldr: "<<hitBuilder;

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
  if(layerNamesInSets.size() == 0)
    theNumberOfLayersInSet = 0;
  else
    theNumberOfLayersInSet = layerNamesInSets[0].size();


  for (IT it = layerNamesInSets.begin(); it != layerNamesInSets.end(); it++) {
    if(it->size() != theNumberOfLayersInSet)
      throw cms::Exception("Configuration") << "Assuming all SeedingLayerSets to have same number of layers. LayerSet " << (it-layerNamesInSets.begin()) << " has " << it->size() << " while 0th has " << theNumberOfLayersInSet;
    for(const std::string& layerName: *it) {
      auto found = std::find(theLayerNames.begin(), theLayerNames.end(), layerName);
      unsigned short layerIndex = 0;
      if(found != theLayerNames.end()) {
        layerIndex = found-theLayerNames.begin();
      }
      else {
        if(std::numeric_limits<unsigned short>::max() == theLayerNames.size()) {
          throw cms::Exception("Assert") << "Too many layers in " << __FILE__ << ":" << __LINE__ << ", we may have to enlarge the index type from unsigned short to unsigned int";
        }

        layerIndex = theLayers.size();
        theLayers.emplace_back(theLayerNames.size(), layerName, layerConfig(layerName, cfg), iC);
        theLayerNames.push_back(layerName);
      }
      theLayerSetIndices.push_back(layerIndex);
    }
  }
  theLayerDets.resize(theLayers.size());
  theTTRHBuilders.resize(theLayers.size());

  // debug printout
  // The following should not be set to cout
  //for(const LayerSpec& layer: theLayers) {
  //  std::cout << layer.print(theLayerNames) << std::endl;
  //}
}

SeedingLayerSetsBuilder::~SeedingLayerSetsBuilder() {}

edm::ParameterSet SeedingLayerSetsBuilder::layerConfig(const std::string & nameLayer,const edm::ParameterSet& cfg) const
{
  edm::ParameterSet result;
   
  for (string::size_type iEnd=nameLayer.size(); iEnd > 0; --iEnd) {
    string name = nameLayer.substr(0,iEnd);
    if (cfg.exists(name)) return cfg.getParameter<edm::ParameterSet>(name);
  } 
  edm::LogError("SeedingLayerSetsBuilder") <<"configuration for layer: "<<nameLayer<<" not found, job will probably crash!";
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

void SeedingLayerSetsBuilder::updateEventSetup(const edm::EventSetup& es) {
  edm::ESHandle<GeometricSearchTracker> htracker;
  es.get<TrackerRecoGeometryRecord>().get( htracker );
  const GeometricSearchTracker& tracker = *htracker;

  const std::vector<BarrelDetLayer const*>&  bpx  = tracker.barrelLayers();
  const std::vector<BarrelDetLayer const*>&  tib  = tracker.tibLayers();
  const std::vector<BarrelDetLayer const*>&  tob  = tracker.tobLayers();

  const std::vector<ForwardDetLayer const*>& fpx_pos = tracker.posForwardLayers();
  const std::vector<ForwardDetLayer const*>& tid_pos = tracker.posTidLayers();
  const std::vector<ForwardDetLayer const*>& tec_pos = tracker.posTecLayers();

  const std::vector<ForwardDetLayer const*>& fpx_neg = tracker.negForwardLayers();
  const std::vector<ForwardDetLayer const*>& tid_neg = tracker.negTidLayers();
  const std::vector<ForwardDetLayer const*>& tec_neg = tracker.negTecLayers();

  for(size_t i=0, n=theLayers.size(); i<n; ++i) {
    const LayerSpec& layer = theLayers[i];
    const DetLayer * detLayer = nullptr;
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
      throw cms::Exception("Configuration") << "Did not find DetLayer for layer " << theLayerNames[layer.nameIndex];
    }

    edm::ESHandle<TransientTrackingRecHitBuilder> builder;
    es.get<TransientRecHitRecord>().get(layer.hitBuilder, builder);

    theLayerDets[i] = detLayer;
    theTTRHBuilders[i] = builder.product();
  }
}

SeedingLayerSets SeedingLayerSetsBuilder::layers(const edm::EventSetup& es)
{
  updateEventSetup(es);

  typedef std::vector<SeedingLayer> Set;
  SeedingLayerSets  result;

  for(size_t i=0, n=theLayerSetIndices.size(); i<n; i += theNumberOfLayersInSet) {
    Set set;
    for(size_t j=0; j<theNumberOfLayersInSet; ++j) {
      const unsigned short layerIndex = theLayerSetIndices[i+j];
      const LayerSpec& layer = theLayers[layerIndex];
      const DetLayer *detLayer = theLayerDets[layerIndex];

      set.push_back( SeedingLayer( theLayerNames[layerIndex], layerIndex, detLayer, theTTRHBuilders[layerIndex], layer.extractor.get()));
    }
    result.push_back(set);
  }
  return result;
}

bool SeedingLayerSetsBuilder::check(const edm::EventSetup& es) {
  // We want to evaluate both in the first invocation (to properly
  // initialize ESWatcher), and this way we avoid one branch compared
  // to || (should be tiny effect)
  return geometryWatcher_.check(es) | trhWatcher_.check(es);
}

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
void
SeedingLayerSetsBuilder::hits(const edm::Event& ev, const edm::EventSetup& es,
			      std::vector<unsigned int> & indices, ctfseeding::SeedingLayer::Hits & hits) const {
  indices.reserve(theLayers.size());
  for(unsigned int i=0; i<theLayers.size(); ++i) {
    // The index of the first hit of this layer
    indices.push_back(hits.size());

    // Obtain and copy the hits
    ctfseeding::SeedingLayer::Hits && tmp = theLayers[i].extractor->hits((const TkTransientTrackingRecHitBuilder &)(*theTTRHBuilders[i]), ev, es);
    std::move(tmp.begin(), tmp.end(), std::back_inserter(hits));
  }
}
