#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"


std::vector<SeedLayerPairs::LayerPair> CosmicLayerPairs::operator()() 
{
  std::vector<SeedLayerPairs::LayerPair> result;

  if (_geometry=="STANDARD"){
//      result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[1], &TIBLayerWithHits[0]));
//      result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[2], &TIBLayerWithHits[0]));

    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[4], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[4]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[7], &TECPlusLayerWithHits[8]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[8]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[7]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[7]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[6]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[6]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[5]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[4]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[3]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[3]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[2]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[0], &TECPlusLayerWithHits[2]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[0], &TECPlusLayerWithHits[1]));


    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[7], &TECMinusLayerWithHits[8]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[6], &TECMinusLayerWithHits[8]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[6], &TECMinusLayerWithHits[7]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[5], &TECMinusLayerWithHits[7]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[5], &TECMinusLayerWithHits[6]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[4], &TECMinusLayerWithHits[6]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[4], &TECMinusLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[3], &TECMinusLayerWithHits[5]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[3], &TECMinusLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[2], &TECMinusLayerWithHits[4]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[2], &TECMinusLayerWithHits[3]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[1], &TECMinusLayerWithHits[3]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[1], &TECMinusLayerWithHits[2]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[0], &TECMinusLayerWithHits[2]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[0], &TECMinusLayerWithHits[1]));


  } 
  else if(_geometry=="TECPAIRS_TOBTRIPLETS"){
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[7], &TECPlusLayerWithHits[8]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[8]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[7]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[7]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[6]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[6]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[5]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[4]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[3]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[3]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[2]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[0], &TECPlusLayerWithHits[2]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[0], &TECPlusLayerWithHits[1]));


    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[7], &TECMinusLayerWithHits[8]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[6], &TECMinusLayerWithHits[8]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[6], &TECMinusLayerWithHits[7]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[5], &TECMinusLayerWithHits[7]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[5], &TECMinusLayerWithHits[6]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[4], &TECMinusLayerWithHits[6]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[4], &TECMinusLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[3], &TECMinusLayerWithHits[5]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[3], &TECMinusLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[2], &TECMinusLayerWithHits[4]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[2], &TECMinusLayerWithHits[3]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[1], &TECMinusLayerWithHits[3]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[1], &TECMinusLayerWithHits[2]));
    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[0], &TECMinusLayerWithHits[2]));

    result.push_back( SeedLayerPairs::LayerPair(&TECMinusLayerWithHits[0], &TECMinusLayerWithHits[1]));
    

  }
  else if (_geometry=="MTCC"){
    result.push_back( SeedLayerPairs::LayerPair(&MTCCLayerWithHits[1],&MTCCLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&MTCCLayerWithHits[2],&MTCCLayerWithHits[3]));
    //IMPORTANT
    // The seed from overlaps must be at the end
    result.push_back( SeedLayerPairs::LayerPair(&MTCCLayerWithHits[0],&MTCCLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&MTCCLayerWithHits[1],&MTCCLayerWithHits[1]));	
  } 
  else if (_geometry=="CRACK"){
    //TODO: clean all this. Now this is a random choice of layers
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[1],&CRACKLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[2],&CRACKLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[2],&CRACKLayerWithHits[1]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[3],&CRACKLayerWithHits[2]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[5],&CRACKLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[6],&CRACKLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[6],&CRACKLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[7],&CRACKLayerWithHits[6]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[8],&CRACKLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[9],&CRACKLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[10],&CRACKLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[11],&CRACKLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[12],&CRACKLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&CRACKLayerWithHits[13],&CRACKLayerWithHits[5]));
  } 
  else if (_geometry=="TIBD+"){
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[1],&TIBLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[2],&TIBLayerWithHits[3]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[0],&TIBLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[1],&TIBLayerWithHits[1]));
  } 
  else if (_geometry=="TOB") {
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[4], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[4]));
    
  } 
  else if(_geometry=="TIBTOB") {
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[1], &TIBLayerWithHits[0]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[2], &TIBLayerWithHits[0]));

    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[4], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[4]));
 
  }
  else if (_geometry=="TEC+") {
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[7], &TECPlusLayerWithHits[8]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[8]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[7]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[7]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[6]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[6]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[5]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[4]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[4]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[3]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[3]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[1]));
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[0]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[0]));
    
  }
  else if (_geometry=="CkfTIBD+"){
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[0], &TIBLayerWithHits[1]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[2], &TIBLayerWithHits[3]));
  } 
  else if (_geometry=="CkfTIBTOB"){
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[0], &TIBLayerWithHits[1]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[0], &TIBLayerWithHits[2]));

    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[4], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[4]));
  }
  else if (_geometry=="CkfTIF3"){
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[0], &TIBLayerWithHits[1]));
    result.push_back( SeedLayerPairs::LayerPair(&TIBLayerWithHits[0], &TIBLayerWithHits[2]));
    
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[4], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[4]));

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[7], &TECPlusLayerWithHits[8]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[8]));	

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[6], &TECPlusLayerWithHits[7]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[7]));
	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[5], &TECPlusLayerWithHits[6]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[6]));	

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[4], &TECPlusLayerWithHits[5]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[5]));	

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[3], &TECPlusLayerWithHits[4]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[4]));	

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[2], &TECPlusLayerWithHits[3]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[3]));	

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[1], &TECPlusLayerWithHits[2]));	
    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[0], &TECPlusLayerWithHits[2]));	

    result.push_back( SeedLayerPairs::LayerPair(&TECPlusLayerWithHits[0], &TECPlusLayerWithHits[1]));	

  } 
  else if (_geometry=="CkfTOB"){
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[0], &TOBLayerWithHits[1]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[0], &TOBLayerWithHits[2]));

    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[4]));	
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[4], &TOBLayerWithHits[5]));
    result.push_back( SeedLayerPairs::LayerPair(&TOBLayerWithHits[3], &TOBLayerWithHits[5]));
  } 
  else {throw cms::Exception("CosmicLayerPairs") << "The geometry " << _geometry << " is not implemented ";}
  return result;
}
CosmicLayerPairs::~CosmicLayerPairs(){}



void CosmicLayerPairs::init(const SiStripRecHit2DCollection &collstereo,
			    const SiStripRecHit2DCollection &collrphi, 
			    const SiStripMatchedRecHit2DCollection &collmatched,
			    //std::string geometry,
			    const edm::EventSetup& iSetup){
    ////std::cout << "initializing geometry " << geometry << std::endl;
    //_geometry=geometry;
  //if(isFirstCall){
    //std::cout << "in isFirtsCall" << std::endl;
    edm::ESHandle<GeometricSearchTracker> track;
    iSetup.get<TrackerRecoGeometryRecord>().get( track );
        //std::cout << "about to take barrel" << std::endl; 
    bl=track->barrelLayers();
	//std::cout << "barrel taken" << std::endl;
    fpos=track->posTecLayers();
	//std::cout << "pos forw taken" << std::endl;
    fneg=track->negTecLayers();		
	//std::cout << "neg forw taken" << std::endl;
  //isFirstCall=false;
    	
    if (_geometry=="MTCC"){//we have to distinguish the MTCC and CRACK case because they have special geometries with different neumbering of layers
	MTCCLayerWithHits.push_back(new LayerWithHits(bl[0], selectTIBHit(collrphi, 1)));
	MTCCLayerWithHits.push_back(new LayerWithHits(bl[1], selectTIBHit(collrphi, 2)));
	MTCCLayerWithHits.push_back(new LayerWithHits(bl[2], selectTOBHit(collrphi, 1)));
	MTCCLayerWithHits.push_back(new LayerWithHits(bl[3], selectTOBHit(collrphi, 2)));
	return;
    }
    if (_geometry=="CRACK"){
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[6], selectTOBHit(collmatched, 7)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[5], selectTOBHit(collmatched, 6)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[3], selectTOBHit(collmatched, 4)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[2], selectTOBHit(collmatched, 3)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[6], selectTOBHit(collrphi, 7)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[5], selectTOBHit(collrphi, 6)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[3], selectTOBHit(collrphi, 4)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[2], selectTOBHit(collrphi, 3)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[4], selectTOBHit(collrphi, 5)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[1], selectTOBHit(collrphi, 2)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[0], selectTOBHit(collrphi, 1)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[4], selectTOBHit(collmatched, 5)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[1], selectTOBHit(collmatched, 2)));
        CRACKLayerWithHits.push_back(new LayerWithHits(bl[0], selectTOBHit(collmatched, 1)));
	return;
    }	
    	
    TIBLayerWithHits.push_back(new LayerWithHits(bl[3], selectTIBHit(collrphi, 1)));  //layer
        //std::cout << "TIB 0" << std::endl;
    TIBLayerWithHits.push_back(new LayerWithHits(bl[4], selectTIBHit(collrphi, 2)));
        //std::cout << "TIB 1" << std::endl;
    TIBLayerWithHits.push_back(new LayerWithHits(bl[5], selectTIBHit(collrphi, 3)));
        //std::cout << "TIB 2" << std::endl;
    TIBLayerWithHits.push_back(new LayerWithHits(bl[6], selectTIBHit(collrphi, 4)));
        //std::cout << "TIB 3" << std::endl;

    TOBLayerWithHits.push_back(new LayerWithHits(bl[7], selectTOBHit(collrphi, 1)));
        //std::cout << "TOB 0" << std::endl;
    TOBLayerWithHits.push_back(new LayerWithHits(bl[8], selectTOBHit(collrphi, 2)));
        //std::cout << "TOB 1" << std::endl;
    TOBLayerWithHits.push_back(new LayerWithHits(bl[9], selectTOBHit(collrphi, 3)));
        //std::cout << "TOB 2" << std::endl;
    TOBLayerWithHits.push_back(new LayerWithHits(bl[10], selectTOBHit(collrphi, 4)));
        //std::cout << "TOB 3" << std::endl;
    TOBLayerWithHits.push_back(new LayerWithHits(bl[11], selectTOBHit(collrphi, 5)));
        //std::cout << "TOB 4" << std::endl;
    TOBLayerWithHits.push_back(new LayerWithHits(bl[12], selectTOBHit(collrphi, 6)));
        //std::cout << "TOB 5" << std::endl;


    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[0], selectTECHit(collrphi, 2, 1)));  //side, disk
	//std::cout << "wheel 0" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[1], selectTECHit(collrphi, 2, 2)));  
	//std::cout << "wheel 1" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[2], selectTECHit(collrphi, 2, 3)));  
	//std::cout << "wheel 2" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[3], selectTECHit(collrphi, 2, 4)));  
	//std::cout << "wheel 3" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[4], selectTECHit(collrphi, 2, 5)));  
	//std::cout << "wheel 4" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[5], selectTECHit(collrphi, 2, 6)));  
	//std::cout << "wheel 5" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[6], selectTECHit(collrphi, 2, 7)));  
	//std::cout << "wheel 6" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[7], selectTECHit(collrphi, 2, 8)));  
	//std::cout << "wheel 7" << std::endl;
    TECPlusLayerWithHits.push_back(new LayerWithHits(fpos[8], selectTECHit(collrphi, 2, 9)));  
	//std::cout << "wheel 8" << std::endl;

    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[0], selectTECHit(collrphi, 1, 1)));  //side, disk
        //std::cout << "wheel 0" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[1], selectTECHit(collrphi, 1, 2)));
        //std::cout << "wheel 1" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[2], selectTECHit(collrphi, 1, 3)));
        //std::cout << "wheel 2" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[3], selectTECHit(collrphi, 1, 4)));
        //std::cout << "wheel 3" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[4], selectTECHit(collrphi, 1, 5)));
        //std::cout << "wheel 4" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[5], selectTECHit(collrphi, 1, 6)));
        //std::cout << "wheel 5" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[6], selectTECHit(collrphi, 1, 7)));
        //std::cout << "wheel 6" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[7], selectTECHit(collrphi, 1, 8)));
        //std::cout << "wheel 7" << std::endl;
    TECMinusLayerWithHits.push_back(new LayerWithHits(fneg[8], selectTECHit(collrphi, 1, 9)));
        //std::cout << "wheel 8" << std::endl;
}

std::vector<const TrackingRecHit*> CosmicLayerPairs::selectTECHit(const SiStripRecHit2DCollection &collrphi,
								int side,
								int disk){
	std::vector<const TrackingRecHit*> theChoosedHits;
  	TrackerLayerIdAccessor acc;
        edmNew::copyDetSetRange(collrphi, theChoosedHits, acc.stripTECDisk(side,disk));
	return theChoosedHits;
	
}

std::vector<const TrackingRecHit*> CosmicLayerPairs::selectTIBHit(const SiStripRecHit2DCollection &collrphi,
                                                                int layer){
	std::vector<const TrackingRecHit*> theChoosedHits;
	TrackerLayerIdAccessor acc; 
        //std::cout << "in selectTIBHit" << std::endl;
        edmNew::copyDetSetRange(collrphi,theChoosedHits,acc.stripTIBLayer(layer));
        return theChoosedHits;

}

std::vector<const TrackingRecHit*> CosmicLayerPairs::selectTOBHit(const SiStripRecHit2DCollection &collrphi,
                                                                int layer){
	std::vector<const TrackingRecHit*> theChoosedHits;
	TrackerLayerIdAccessor acc;
        //std::cout << "in selectTOBHit" << std::endl;
        edmNew::copyDetSetRange(collrphi,theChoosedHits,acc.stripTOBLayer(layer));
        return theChoosedHits;
}

std::vector<const TrackingRecHit*> CosmicLayerPairs::selectTECHit(const SiStripMatchedRecHit2DCollection &collmatch,
                                                                int side,
                                                                int disk){
        std::vector<const TrackingRecHit*> theChoosedHits;
        TrackerLayerIdAccessor acc;
        //std::cout << "in selectTECHit" << std::endl;
        edmNew::copyDetSetRange(collmatch,theChoosedHits,acc.stripTECDisk(side,disk));
        return theChoosedHits;

}

std::vector<const TrackingRecHit*> CosmicLayerPairs::selectTIBHit(const SiStripMatchedRecHit2DCollection &collmatch,
                                                                int layer){
        std::vector<const TrackingRecHit*> theChoosedHits;
        TrackerLayerIdAccessor acc;
        //std::cout << "in selectTIBHit" << std::endl;
        edmNew::copyDetSetRange(collmatch,theChoosedHits,acc.stripTIBLayer(layer));
        return theChoosedHits;

}

std::vector<const TrackingRecHit*> CosmicLayerPairs::selectTOBHit(const SiStripMatchedRecHit2DCollection &collmatch,
                                                                int layer){
        std::vector<const TrackingRecHit*> theChoosedHits;
        TrackerLayerIdAccessor acc;
        //std::cout << "in selectTOBHit" << std::endl;
        edmNew::copyDetSetRange(collmatch,theChoosedHits,acc.stripTOBLayer(layer));
        return theChoosedHits;
}
