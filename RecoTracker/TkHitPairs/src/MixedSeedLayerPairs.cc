#include "RecoTracker/TkHitPairs/interface/MixedSeedLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

using std::vector;

MixedSeedLayerPairs::~MixedSeedLayerPairs(){
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
}

vector<SeedLayerPairs::LayerPair> MixedSeedLayerPairs::operator()() 
{
  vector<LayerPair> result;
  
  //seeds from the barrel
  addBarrelBarrelLayers(0,1,result);

  addBarrelBarrelLayers(0,2,result);  
  addBarrelBarrelLayers(1,2,result);

  //seeds from the forward-barrel
  addBarrelForwardLayers(0,0,result);
  addBarrelForwardLayers(0,1,result);
  addBarrelForwardLayers(1,0,result);
  addBarrelForwardLayers(1,1,result);

  //seeds from the forward
  addForwardForwardLayers(0,1,result);
  //addForwardForwardLayers(0,2,result); Overlaps (0,1) --> redundant
  //addForwardForwardLayers(0,3,result); Overlaps (0,1) --> redundant
  //addForwardForwardLayers(0,4,result); Overlaps (0,1) --> redundant

  addForwardForwardLayers(1,2,result);
  addForwardForwardLayers(1,3,result);
  //addForwardForwardLayers(1,4,result); Overlaps (0,1) --> redundant

  //addForwardForwardLayers(2,3,result); Overlaps (1,2) --> redundant

  addForwardForwardLayers(3,4,result);



  LogDebug("TkHitPairs") << "Mixed layersPair.size: " << result.size() ;
  return result;
}


void MixedSeedLayerPairs::init(const SiPixelRecHitCollection &collpxl,
			       const SiStripMatchedRecHit2DCollection &collmatch,
			       const SiStripRecHit2DCollection &collstereo, 
			       const SiStripRecHit2DCollection &collrphi, 
			       const edm::EventSetup& iSetup)
{
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
  allLayersWithHits.clear();
  barrelLayers.clear();
  fwdLayers.clear();
  bkwLayers.clear();

  if(isFirstCall){
    edm::ESHandle<GeometricSearchTracker> track;
    iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
    detLayersPxlBarrel = track->pixelBarrelLayers();
    detLayersPosPxl    = track->posPixelForwardLayers();
    detLayersNegPxl    = track->negPixelForwardLayers();
    detLayersTIB    = track->tibLayers();
    detLayersPosTID = track->posTidLayers();
    detLayersNegTID = track->negTidLayers();
    detLayersPosTEC = track->posTecLayers();
    detLayersNegTEC = track->negTecLayers();
    isFirstCall=false;
  }
  
  SiPixelRecHitCollection::range map_range1=collpxl.get(acc.pixelBarrelLayer(1));
  SiPixelRecHitCollection::range map_range2=collpxl.get(acc.pixelBarrelLayer(2));
  SiPixelRecHitCollection::range map_range3=collpxl.get(acc.pixelBarrelLayer(3));

  SiPixelRecHitCollection::range map_diskpos1=collpxl.get(acc.pixelForwardDisk(2,1));
  SiPixelRecHitCollection::range map_diskpos2=collpxl.get(acc.pixelForwardDisk(2,2));
  SiPixelRecHitCollection::range map_diskneg1=collpxl.get(acc.pixelForwardDisk(1,1));
  SiPixelRecHitCollection::range map_diskneg2=collpxl.get(acc.pixelForwardDisk(1,2));

  
  barrelLayers.push_back(new LayerWithHits(detLayersPxlBarrel[0],map_range1) );
  barrelLayers.push_back(new LayerWithHits(detLayersPxlBarrel[1],map_range2) );
  barrelLayers.push_back(new LayerWithHits(detLayersPxlBarrel[2],map_range3) );

  fwdLayers.push_back(new LayerWithHits(detLayersPosPxl[0],map_diskpos1) );
  fwdLayers.push_back(new LayerWithHits(detLayersPosPxl[1],map_diskpos2) );
  bkwLayers.push_back(new LayerWithHits(detLayersNegPxl[0],map_diskneg1) );
  bkwLayers.push_back(new LayerWithHits(detLayersNegPxl[1],map_diskneg2) );

			 
  /*
  //TID 2=forward, 1,2,3=disk, 1= firstRing 2,3=lastRing  
  fwdLayers.push_back(new LayerWithHits(detLayersPosTID[0],selectHitTID(collmatch,
									collstereo,
									collrphi,
									2,1,1,2      )));   

  fwdLayers.push_back(new LayerWithHits(detLayersPosTID[1],selectHitTID(collmatch,
									collstereo,
									collrphi,
									2,2,1,3      )));   
  
  fwdLayers.push_back(new LayerWithHits(detLayersPosTID[2],selectHitTID(collmatch,
									collstereo,
									collrphi,
									2,3,1,2      )));   

  //TID 1=backward, 1,2,3=disk, 1=firstRing 2,3=lastRing
  
  bkwLayers.push_back(new LayerWithHits(detLayersNegTID[0],selectHitTID(collmatch,
									collstereo,
									collrphi,
									1,1,1,2      )));   

  bkwLayers.push_back(new LayerWithHits(detLayersNegTID[1],selectHitTID(collmatch,
									collstereo,
									collrphi,
									1,2,1,3      )));   
  
  bkwLayers.push_back(new LayerWithHits(detLayersNegTID[2],selectHitTID(collmatch,
									collstereo,
									collrphi,
									1,3,1,2      )));   
  
  
  */

  //TEC 2=forward, 1,2,3=disk, 1=firstRing 2=lastRing
  fwdLayers.push_back(new LayerWithHits(detLayersPosTEC[1],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									2,1,1,2      ))); 

  fwdLayers.push_back(new LayerWithHits(detLayersPosTEC[1],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									2,2,1,2      ))); 

  fwdLayers.push_back(new LayerWithHits(detLayersPosTEC[2],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									2,3,1,2      )));   

  //TEC 1=backward, 1,2,3=disk, 1=firstRing 2=lastRing
  bkwLayers.push_back(new LayerWithHits(detLayersNegTEC[1],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									1,1,1,2      ))); 

  bkwLayers.push_back(new LayerWithHits(detLayersNegTEC[1],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									1,2,1,2      ))); 

  bkwLayers.push_back(new LayerWithHits(detLayersNegTEC[2],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									1,3,1,2      )));   
  
  
  allLayersWithHits.insert(allLayersWithHits.end(),barrelLayers.begin(),barrelLayers.end());
  allLayersWithHits.insert(allLayersWithHits.end(),fwdLayers.begin(),fwdLayers.end());
  allLayersWithHits.insert(allLayersWithHits.end(),bkwLayers.begin(),bkwLayers.end());
}


vector<const TrackingRecHit*> 
MixedSeedLayerPairs::selectHitTIB(const SiStripMatchedRecHit2DCollection &collmatch,
				      const SiStripRecHit2DCollection &collstereo, 
				      const SiStripRecHit2DCollection &collrphi,
				      int tibNumber) {  
  vector<const TrackingRecHit*> theChoosedHits;

  
  SiStripMatchedRecHit2DCollection::range range2D = collmatch.get(acc.stripTIBLayer(tibNumber));
  for(SiStripMatchedRecHit2DCollection::const_iterator it = range2D.first;
      it != range2D.second; it++){
    theChoosedHits.push_back( &(*it) );
  }
  
  
  /*  
  SiStripRecHit2DCollection::range rangeRphi = collrphi.get(acc.stripTIBLayer(tibNumber));
  for(SiStripRecHit2DCollection::const_iterator it = rangeRphi.first;
      it != rangeRphi.second; it++){
    //add a filter to avoid double counting rphi hit of matcherRecHit
    theChoosedHits.push_back( &(*it) );
  }


  SiStripRecHit2DCollection::range rangeStereo = collstereo.get(acc.stripTIBLayer(tibNumber));
  for(SiStripRecHit2DCollection::const_iterator it = rangeStereo.first;
      it != rangeStereo.second; it++){
    //add a filter to avoid double counting stereo hit of matcherRecHit
    theChoosedHits.push_back( &(*it) );
  }  
  */
  
  LogDebug("TkHitPairs") << "size choosed hits for TIB layer: " << theChoosedHits.size() ;
  return theChoosedHits;
}


vector<const TrackingRecHit*> 
MixedSeedLayerPairs::selectHitTID(const SiStripMatchedRecHit2DCollection &collmatch,
				      const SiStripRecHit2DCollection &collstereo, 
				      const SiStripRecHit2DCollection &collrphi,
				      int side,
				      int disk,
				      int firstRing,
				      int lastRing)
{
  vector<const TrackingRecHit*> theChoosedHits;
  
  SiStripMatchedRecHit2DCollection::range range2D = collmatch.get(acc.stripTIDDisk(side,disk));
  for(SiStripMatchedRecHit2DCollection::const_iterator it = range2D.first;
      it != range2D.second; it++){
    int ring = TIDDetId( it->geographicalId() ).ring();
    if(ring>=firstRing && ring<=lastRing) theChoosedHits.push_back( &(*it) );
  }
  
  
  SiStripRecHit2DCollection::range rangeRphi = collrphi.get(acc.stripTIDDisk(side,disk));
  for(SiStripRecHit2DCollection::const_iterator it = rangeRphi.first;
      it != rangeRphi.second; it++){
    //add a filter to avoid double counting rphi hit of matcherRecHit
    int ring = TIDDetId( it->geographicalId() ).ring();
    // ---- until the filter over rphi hit is not implemented,
    // the first two rings are skipped
    //if(ring>=firstRing && ring<=lastRing) theChoosedHits.push_back( &(*it) );
    if(ring>=3 && ring<=lastRing) theChoosedHits.push_back( &(*it) );

  }

  /*
  SiStripRecHit2DCollection::range rangeStereo = collstereo.get(acc.stripTIDDisk(side,disk));
  for(SiStripRecHit2DCollection::const_iterator it = rangeStereo.first;
      it != rangeStereo.second; it++){
    //add a filter to avoid double counting stereo hit of matcherRecHit
    theChoosedHits.push_back( &(*it) );
  }
  */

  LogDebug("TkHitPairs") << "size choosed hits for TID layer: " << theChoosedHits.size() ;
  return theChoosedHits;
}


vector<const TrackingRecHit*> 
MixedSeedLayerPairs::selectHitTEC(const SiStripMatchedRecHit2DCollection &collmatch,
				      const SiStripRecHit2DCollection &collstereo, 
				      const SiStripRecHit2DCollection &collrphi,
				      int side,
				      int disk,
				      int firstRing,
				      int lastRing)
{  
  vector<const TrackingRecHit*> theChoosedHits;
  
  
  SiStripMatchedRecHit2DCollection::range range2D = collmatch.get(acc.stripTECDisk(side,disk));
  for(SiStripMatchedRecHit2DCollection::const_iterator it = range2D.first;
      it != range2D.second; it++){
    int ring = TECDetId( it->geographicalId() ).ring();
    if(ring>=firstRing && ring<=lastRing) theChoosedHits.push_back( &(*it) );
  }
  
  
  
  /*
  SiStripRecHit2DCollection::range rangeRphi = collrphi.get(acc.stripTECDisk(side,disk));
  for(SiStripRecHit2DCollection::const_iterator it = rangeRphi.first;
      it != rangeRphi.second; it++){
    //add a filter to avoid double counting rphi hit of matcherRecHit
    theChoosedHits.push_back( &(*it) );
  }
  

  
  SiStripRecHit2DCollection::range rangeStereo = collstereo.get(acc.stripTECDisk(side,disk));
  for(SiStripRecHit2DCollection::const_iterator it = rangeStereo.first;
      it != rangeStereo.second; it++){
    //add a filter to avoid double counting stereo hit of matcherRecHit
    theChoosedHits.push_back( &(*it) );
  }
  */

  return theChoosedHits;
}


void MixedSeedLayerPairs::addBarrelBarrelLayers( int mid, int outer, 
						 vector<LayerPair>& result) const
{
  result.push_back( LayerPair( barrelLayers[mid], barrelLayers[outer]));
}



void MixedSeedLayerPairs::addBarrelForwardLayers( int mid, int outer, 
						  vector<LayerPair>& result) const
{
  result.push_back( LayerPair( barrelLayers[mid], bkwLayers[outer]));
  result.push_back( LayerPair( barrelLayers[mid], fwdLayers[outer]));
}


void MixedSeedLayerPairs::addForwardForwardLayers( int mid, int outer, 
						   vector<LayerPair>& result) const
{
  result.push_back( LayerPair( bkwLayers[mid], bkwLayers[outer]));
  result.push_back( LayerPair( fwdLayers[mid], fwdLayers[outer]));
}
