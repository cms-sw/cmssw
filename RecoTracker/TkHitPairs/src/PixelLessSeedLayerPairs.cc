#include "RecoTracker/TkHitPairs/interface/PixelLessSeedLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

using std::vector;

PixelLessSeedLayerPairs::~PixelLessSeedLayerPairs(){
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
}

vector<SeedLayerPairs::LayerPair> PixelLessSeedLayerPairs::operator()() 
{
  vector<LayerPair> result;
  
  addBarrelBarrelLayers(0,1,result);
  
  addBarrelForwardLayers(0,0,result);
  addBarrelForwardLayers(0,1,result);
  addBarrelForwardLayers(0,2,result);

  addBarrelForwardLayers(1,0,result);
  addBarrelForwardLayers(1,1,result);
  addBarrelForwardLayers(1,2,result);


  addForwardForwardLayers(0,1,result);
  addForwardForwardLayers(0,2,result);
  addForwardForwardLayers(0,3,result);
  addForwardForwardLayers(0,4,result);

  addForwardForwardLayers(1,2,result);
  addForwardForwardLayers(1,3,result);
  addForwardForwardLayers(1,4,result);

  addForwardForwardLayers(2,3,result);
  addForwardForwardLayers(2,4,result);

  addForwardForwardLayers(3,4,result);
  

  LogDebug("TkHitPairs") << "PixelLess layersPair.size: " << result.size() ;
  return result;
}


void PixelLessSeedLayerPairs::init(const SiStripMatchedRecHit2DCollection &collmatch,
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
    detLayersTIB    = track->tibLayers();
    detLayersPosTID = track->posTidLayers();
    detLayersNegTID = track->negTidLayers();
    detLayersPosTEC = track->posTecLayers();
    detLayersNegTEC = track->negTecLayers();
    isFirstCall=false;
  }
  
  
  

  barrelLayers.push_back(new LayerWithHits(detLayersTIB[0],selectHitTIB(collmatch,collstereo,collrphi,1)) );
  barrelLayers.push_back(new LayerWithHits(detLayersTIB[1],selectHitTIB(collmatch,collstereo,collrphi,2)) );
  
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
  

  //TEC 2=forward, 2,3=disk, 1=firstRing 2=lastRing
  fwdLayers.push_back(new LayerWithHits(detLayersPosTEC[1],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									2,2,1,2      ))); 

  fwdLayers.push_back(new LayerWithHits(detLayersPosTEC[2],selectHitTEC(collmatch,
									collstereo,
									collrphi,
									2,3,1,2      )));   

  //TEC 1=backward, 2,3=disk, 1=firstRing 2=lastRing
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
PixelLessSeedLayerPairs::selectHitTIB(const SiStripMatchedRecHit2DCollection &collmatch,
				      const SiStripRecHit2DCollection &collstereo, 
				      const SiStripRecHit2DCollection &collrphi,
				      int tibNumber) {  
  vector<const TrackingRecHit*> theChoosedHits;

  
  SiStripMatchedRecHit2DCollection::range range2D = collmatch.get(acc.stripTIBLayer(tibNumber));
  for(SiStripMatchedRecHit2DCollection::const_iterator it = range2D.first;
      it != range2D.second; it++){
    theChoosedHits.push_back( &(*it) );
  }
  
  
    
  SiStripRecHit2DCollection::range rangeRphi = collrphi.get(acc.stripTIBLayer(tibNumber));
  for(SiStripRecHit2DCollection::const_iterator it = rangeRphi.first;
      it != rangeRphi.second; it++){
    //add a filter to avoid double counting rphi hit of matcherRecHit
    theChoosedHits.push_back( &(*it) );
  }
  
  /*
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
PixelLessSeedLayerPairs::selectHitTID(const SiStripMatchedRecHit2DCollection &collmatch,
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
PixelLessSeedLayerPairs::selectHitTEC(const SiStripMatchedRecHit2DCollection &collmatch,
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


void PixelLessSeedLayerPairs::addBarrelBarrelLayers( int mid, int outer, 
						     vector<LayerPair>& result) const
{
  result.push_back( LayerPair( barrelLayers[mid], barrelLayers[outer]));
}



void PixelLessSeedLayerPairs::addBarrelForwardLayers( int mid, int outer, 
						      vector<LayerPair>& result) const
{
  result.push_back( LayerPair( barrelLayers[mid], bkwLayers[outer]));
  result.push_back( LayerPair( barrelLayers[mid], fwdLayers[outer]));
}


void PixelLessSeedLayerPairs::addForwardForwardLayers( int mid, int outer, 
						       vector<LayerPair>& result) const
{
  result.push_back( LayerPair( bkwLayers[mid], bkwLayers[outer]));
  result.push_back( LayerPair( fwdLayers[mid], fwdLayers[outer]));
}
