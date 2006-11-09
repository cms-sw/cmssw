#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"

vector<SeedLayerPairs::LayerPair> CosmicLayerPairs::operator()() 
{
  vector<LayerPair> result;

  if (_geometry=="STANDARD"){
    result.push_back( LayerPair(lh1,lh2));
    result.push_back( LayerPair(lh2,lh3));
    result.push_back( LayerPair(lh1,lh3));
  }

  if (_geometry=="MTCC"){
 
    result.push_back( LayerPair(lh2,lh1));
    result.push_back( LayerPair(lh3,lh4));
    //IMPORTANT
    // The seed from overlaps must be at the end
    result.push_back( LayerPair(lh1,lh1));
    result.push_back( LayerPair(lh2,lh2));
  }

  if (_geometry=="CRACK"){
    result.push_back( LayerPair(lh2,lh1));
//    result.push_back( LayerPair(lh3,lh1));
//    result.push_back( LayerPair(lh3,lh2));
  }

  return result;
}
CosmicLayerPairs::~CosmicLayerPairs()
{
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
}



void CosmicLayerPairs::init(const SiStripRecHit2DCollection &collstereo,
			    const SiStripRecHit2DCollection &collrphi, 
			    const SiStripMatchedRecHit2DCollection &collmatched,
			    std::string geometry,
			    const edm::EventSetup& iSetup){
  _geometry=geometry;
  if(isFirstCall){
    edm::ESHandle<GeometricSearchTracker> track;
    iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
    bl=track->barrelLayers(); 
    isFirstCall=false;
  }
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); 
      it!=allLayersWithHits.end();it++){
    delete *it;
  }

  allLayersWithHits.clear();
  LogDebug("CosmicSeedFinder") <<"Reconstruction for geometry  "<<_geometry;
  if (_geometry=="STANDARD"){
    rphi_range1=collrphi.get(acc.stripTOBLayer(4));
    rphi_range2=collrphi.get(acc.stripTOBLayer(5));
    rphi_range3=collrphi.get(acc.stripTOBLayer(6));
    
    const TOBLayer*  bl1=dynamic_cast<TOBLayer*>(bl[10]);
    const TOBLayer*  bl2=dynamic_cast<TOBLayer*>(bl[11]);
    const TOBLayer*  bl3=dynamic_cast<TOBLayer*>(bl[12]);
    
  

    //   //LayersWithHits
    lh1=new  LayerWithHits(bl1,rphi_range1);   allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2);   allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3);   allLayersWithHits.push_back(lh3);
  }
  if(_geometry=="MTCC"){ 

    const TIBLayer*  bl1=dynamic_cast<TIBLayer*>(bl[0]);
    const TIBLayer*  bl2=dynamic_cast<TIBLayer*>(bl[1]);
    const TOBLayer*  bl3=dynamic_cast<TOBLayer*>(bl[2]);
    const TOBLayer*  bl4=dynamic_cast<TOBLayer*>(bl[3]);
    rphi_range1=collrphi.get(acc.stripTIBLayer(1));
    rphi_range2=collrphi.get(acc.stripTIBLayer(2));
    rphi_range3=collrphi.get(acc.stripTOBLayer(1));
    rphi_range4=collrphi.get(acc.stripTOBLayer(2));

    lh1=new  LayerWithHits(bl1,rphi_range1); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,rphi_range4); allLayersWithHits.push_back(lh4);
  }


  if (_geometry=="CRACK"){

    LogDebug("CosmicSeedFinder") <<" Using CRack seeding "<< _geometry << endl;
    match_range1=collmatched.get(acc.stripTOBLayer(7));
    rphi_range2=collrphi.get(acc.stripTOBLayer(6));
    match_range3=collmatched.get(acc.stripTOBLayer(4));
    
    const TOBLayer*  bl1=dynamic_cast<TOBLayer*>(bl[6]);
    const TOBLayer*  bl2=dynamic_cast<TOBLayer*>(bl[5]);
    const TOBLayer*  bl3=dynamic_cast<TOBLayer*>(bl[3]);
    
    
    //LayersWithHits
    lh1=new  LayerWithHits(bl1,match_range1);
    lh2=new  LayerWithHits(bl2,rphi_range2);
    lh3=new  LayerWithHits(bl3,match_range3);
    
  }
}
