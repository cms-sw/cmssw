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

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"  

using std::vector;

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
  if (_geometry=="TIBD+"){
 
    result.push_back( LayerPair(lh2,lh1));
    result.push_back( LayerPair(lh3,lh4));
    result.push_back( LayerPair(lh1,lh1));
    result.push_back( LayerPair(lh2,lh2));
  }
  if (_geometry=="CkfTIBD+"){

    result.push_back( LayerPair(lh1,lh2));
    result.push_back( LayerPair(lh3,lh4));
  }
  if (_geometry=="CkfTIBD+3"){

    result.push_back( LayerPair(lh1,lh2));
    result.push_back( LayerPair(lh1,lh3));	
    result.push_back( LayerPair(lh3,lh4));
    result.push_back( LayerPair(lh2,lh4));
  }
  if (_geometry=="CkfTIF"){
    result.push_back( LayerPair(lh1,lh2));
    result.push_back( LayerPair(lh3,lh4));		
  }
  if (_geometry=="CkfTIF3"){
    result.push_back( LayerPair(lh1,lh2));
    result.push_back( LayerPair(lh1,lh3));
    result.push_back( LayerPair(lh4,lh5));
    result.push_back( LayerPair(lh5,lh6));
    result.push_back( LayerPair(lh4,lh6));
  }
  if (_geometry=="CkfTOB"){
    result.push_back( LayerPair(lh1,lh2));
    result.push_back( LayerPair(lh1,lh3));
    result.push_back( LayerPair(lh4,lh5));
    result.push_back( LayerPair(lh5,lh6));
    result.push_back( LayerPair(lh4,lh6));
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

    LogDebug("CosmicSeedFinder") <<" Using CRack seeding "<< _geometry << std::endl;
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
  if (_geometry=="TIBD+"){
  
    const TIBLayer*  bl1=dynamic_cast<TIBLayer*>(bl[3]);
    const TIBLayer*  bl2=dynamic_cast<TIBLayer*>(bl[4]);
    const TIBLayer*  bl3=dynamic_cast<TIBLayer*>(bl[5]);
    const TIBLayer*  bl4=dynamic_cast<TIBLayer*>(bl[6]);
    rphi_range1=collrphi.get(acc.stripTIBLayer(1));
    rphi_range2=collrphi.get(acc.stripTIBLayer(2));
    rphi_range3=collrphi.get(acc.stripTIBLayer(3));
    rphi_range4=collrphi.get(acc.stripTIBLayer(4));
    lh1=new  LayerWithHits(bl1,rphi_range1); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,rphi_range4); allLayersWithHits.push_back(lh4);

  }
  if ((_geometry=="CkfTIBD+") || (_geometry=="CkfTIBD+3")){

    const TIBLayer*  bl1=dynamic_cast<TIBLayer*>(bl[3]);
    const TIBLayer*  bl2=dynamic_cast<TIBLayer*>(bl[4]);
    const TIBLayer*  bl3=dynamic_cast<TIBLayer*>(bl[5]);
    const TIBLayer*  bl4=dynamic_cast<TIBLayer*>(bl[6]);
    rphi_range1=collrphi.get(acc.stripTIBLayer(1));
    rphi_range2=collrphi.get(acc.stripTIBLayer(2));
    rphi_range3=collrphi.get(acc.stripTIBLayer(3));
    rphi_range4=collrphi.get(acc.stripTIBLayer(4));
/*
    SiStripRecHit2DCollection myl1collection;
    SiStripRecHit2DCollection myl2collection;	
    SiStripRecHit2DCollection::const_iterator irphi = collrphi.begin();
    
    for (; irphi != collrphi.end(); irphi ++){
	DetId id = irphi->geographicalId();
	if (StripSubdetector(id).subdetId() == StripSubdetector::TIB){
		TIBDetId tibid = TIBDetId(id);
		std::cout << "Inserting " << id.rawId() << std::endl; 
		if (tibid.layer()==1) myl1collection.put(id, collrphi.get(id).first, collrphi.get(id).second);
		if (tibid.layer()==2) myl2collection.put(id, collrphi.get(id).first, collrphi.get(id).second);
	}
    }	 
    rphi_range1.first = myl1collection.begin(); rphi_range1.second = myl1collection.end();
    rphi_range2.first = myl2collection.begin(); rphi_range2.second = myl2collection.end();
*/
    lh1=new  LayerWithHits(bl1,rphi_range1); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,rphi_range4); allLayersWithHits.push_back(lh4);	
  }
  

  if (_geometry=="CkfTIF"){
    const TIBLayer*  bl1=dynamic_cast<TIBLayer*>(bl[3]);
    const TIBLayer*  bl2=dynamic_cast<TIBLayer*>(bl[4]);
    const TOBLayer*  bl3=dynamic_cast<TOBLayer*>(bl[11]);
    const TOBLayer*  bl4=dynamic_cast<TOBLayer*>(bl[12]);
    rphi_range1=collrphi.get(acc.stripTIBLayer(1));
    rphi_range2=collrphi.get(acc.stripTIBLayer(2));
    rphi_range3=collrphi.get(acc.stripTOBLayer(5));
    rphi_range4=collrphi.get(acc.stripTOBLayer(6));
    
    lh1=new  LayerWithHits(bl1,rphi_range1); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,rphi_range4); allLayersWithHits.push_back(lh4);	

  }	
  if (_geometry=="CkfTIF3"){
    const TIBLayer*  bl1=dynamic_cast<TIBLayer*>(bl[3]);
    const TIBLayer*  bl2=dynamic_cast<TIBLayer*>(bl[4]);
    const TIBLayer*  bl3=dynamic_cast<TIBLayer*>(bl[5]);	
    const TOBLayer*  bl4=dynamic_cast<TOBLayer*>(bl[10]);	
    const TOBLayer*  bl5=dynamic_cast<TOBLayer*>(bl[11]);
    const TOBLayer*  bl6=dynamic_cast<TOBLayer*>(bl[12]);
    rphi_range1=collrphi.get(acc.stripTIBLayer(1));
    rphi_range2=collrphi.get(acc.stripTIBLayer(2));
    rphi_range3=collrphi.get(acc.stripTIBLayer(3));
    rphi_range4=collrphi.get(acc.stripTOBLayer(4));
    rphi_range5=collrphi.get(acc.stripTOBLayer(5));
    rphi_range6=collrphi.get(acc.stripTOBLayer(6));	

    lh1=new  LayerWithHits(bl1,rphi_range1); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,rphi_range4); allLayersWithHits.push_back(lh4);
    lh5=new  LayerWithHits(bl5,rphi_range5); allLayersWithHits.push_back(lh5);
    lh6=new  LayerWithHits(bl6,rphi_range6); allLayersWithHits.push_back(lh6);
    

  }
  if (_geometry=="CkfTOB"){
    const TOBLayer*  bl1=dynamic_cast<TOBLayer*>(bl[7]);
    const TOBLayer*  bl2=dynamic_cast<TOBLayer*>(bl[8]);
    const TOBLayer*  bl3=dynamic_cast<TOBLayer*>(bl[9]);
    const TOBLayer*  bl4=dynamic_cast<TOBLayer*>(bl[10]);
    const TOBLayer*  bl5=dynamic_cast<TOBLayer*>(bl[11]);
    const TOBLayer*  bl6=dynamic_cast<TOBLayer*>(bl[12]);
    rphi_range1=collrphi.get(acc.stripTOBLayer(1));
    rphi_range2=collrphi.get(acc.stripTOBLayer(2));
    rphi_range3=collrphi.get(acc.stripTOBLayer(3));
    rphi_range4=collrphi.get(acc.stripTOBLayer(4));
    rphi_range5=collrphi.get(acc.stripTOBLayer(5));
    rphi_range6=collrphi.get(acc.stripTOBLayer(6));

    lh1=new  LayerWithHits(bl1,rphi_range1); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,rphi_range2); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,rphi_range3); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,rphi_range4); allLayersWithHits.push_back(lh4);
    lh5=new  LayerWithHits(bl5,rphi_range5); allLayersWithHits.push_back(lh5);
    lh6=new  LayerWithHits(bl6,rphi_range6); allLayersWithHits.push_back(lh6);


  }

}
