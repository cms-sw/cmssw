///////////////////////////////////////////////////////////////////////////////
// File: DigiCollectionFP420.cc
// Date: 12.2006
// Description: DigiCollectionFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
//
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
//#include "SimRomanPot/SimFP420/interface/FP420DigiMain.h"
#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"

#include "SimRomanPot/SimFP420/interface/ClusterFP420.h"

#include <cstdlib> 


#include <vector>

using namespace std;

//#include <iostream>


//#define mydigidebug0

//namespace fp420
//{

DigitizerFP420::DigitizerFP420(const edm::ParameterSet& conf):conf_(conf),stripDigitizer_(new FP420DigiMain(conf)) {
    
  edm::ParameterSet m_Anal = conf.getParameter<edm::ParameterSet>("DigitizerFP420");
    verbosity    = m_Anal.getParameter<int>("Verbosity");
    sn0 = m_Anal.getParameter<int>("NumberFP420Stations");
    pn0 = m_Anal.getParameter<int>("NumberFP420SPlanes");
    if(verbosity>0) {
      std::cout << "Creating a DigitizerFP420" << std::endl;
      std::cout << "DigitizerFP420: sn0=" << sn0 << " pn0=" << pn0 << std::endl;
    }
  
}

// Virtual destructor needed.
DigitizerFP420::~DigitizerFP420() { 
  if(verbosity>0) {
    std::cout << "Destroying a DigitizerFP420" << std::endl;
  }
  delete stripDigitizer_;
  
}  

// Functions that gets called by framework every event
void DigitizerFP420::produce(FP420G4HitCollection *   theCAFI, DigiCollectionFP420 & output) {
  // Step A: Get Inputs
  theStripHits.clear();
  
  //Loop on FP420G4Hit
  SimHitMap.clear();

  // hit map for FP420
  // ==================================
  map<int,float,less<int> > themap;
  
  for (int j=0; j<theCAFI->entries(); j++) {
    FP420G4Hit* isim = (*theCAFI)[j];
    
    unsigned int unitID = (*isim).getUnitID();
    int det, zside, sector, zmodule, sScale = 2*(pn0-1);
    FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
    // intindex is a continues numbering of FP420
    int zScale=2;  unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
    //	 int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
#ifdef mydigidebug0
    std::cout << " DigitizerFP420:1 start hitcoll   index = " <<  unitID <<  " iu = " << intindex  << std::endl;
#endif
    double  losenergy = isim->getEnergyLoss();
    themap[unitID] += losenergy;
#ifdef mydigidebug0
    Hep3Vector hitPoint = isim->getEntry();
    float   tof = isim->getTof();
    std::cout <<"DigitizerFP420:  unitID = " << unitID << "  tof= " << tof <<"  intindex= " << intindex << "  losenergy= " << losenergy << "  sector= " << sector << "  zmodule= " << zmodule << "  zside= " << zside << "  hitPoint= " << hitPoint << std::endl;
#endif
    // push all hit collection pointers (*isim) variables
    SimHitMap[intindex].push_back((*isim));
    
  }
  
  //============================================================================================================================
  //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!   
  // clean info of container from previous event
  for (int sector=1; sector<sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=1; zside<3; zside++) {
	int sScale = 2*(pn0-1);
	//      int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	// intindex is a continues numbering of FP420
	int zScale=2;  unsigned int detID = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	std::vector<HDigiFP420> collector;
	collector.clear();
	DigiCollectionFP420::Range inputRange;
	inputRange.first = collector.begin();
	inputRange.second = collector.end();
	output.putclear(inputRange,detID);
      }//for
    }//for
  }//for
  //                                                                                                                      !!!!!!   
  // if we want to keep Digi container/Collection for one event uncomment the line below and vice versa
  output.clear();   //container_.clear() --> start from the beginning of the container
  //============================================================================================================================================
  
  
  bool first = true;
  for (int sector=1; sector<sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=1; zside<3; zside++) {
	
	int sScale = 2*(pn0-1), det= 1;
	//      int index = CaloNumberingPacker::packCastorIndex(det,zside, sector, zmodule);
	int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	double   theTotalEnergy = themap[index];
	//	if(theTotalEnergy <= 0.00003) break;
	if(theTotalEnergy <= 0.00000) break;
	
	// intindex is a continues numbering of FP420
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	// int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
#ifdef mydigidebug0
	std::cout << " DigitizerFP420:2 run loop   index = " <<  index <<  " iu = " << iu  << std::endl;
#endif
	
	//    GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
	//      CLHEP::Hep3Vector Bfieldloc=bfield();
  	G4ThreeVector bfield(  0.,  0.,  0.0  );
	//	G4ThreeVector bfield(  0.5,  0.5,  1.0  );
	
	
#ifdef mydigidebug0
	std::cout <<" ===" << std::endl;
	std::cout <<" ============== DigitizerFP420:  call run for iu= " << iu << std::endl;
	std::cout <<" ===" << std::endl;
#endif
	collector.clear();
//	collector= stripDigitizer_.run( SimHitMap[iu],
//					bfield,
//					iu,
//					sScale
//					); // stripDigitizer_.run...  return 


	collector= stripDigitizer_->run( SimHitMap[iu],
					bfield,
					iu,
					sScale
					); // stripDigitizer_.run...  return 



#ifdef mydigidebug0
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<"=======  DigitizerFP420:  collector size = " << collector.size() << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
#endif
	
	
	
	//	if (collector.size()>0){
#ifdef mydigidebug0
	  std::cout <<"         ============= DigitizerFP420:collector start!!!!!!!!!!!!!!" << std::endl;
#endif
	  DigiCollectionFP420::Range outputRange;
	  outputRange.first = collector.begin();
	  outputRange.second = collector.end();
	  
	  if ( first ) {
	    // use it only if ClusterCollectionFP420 is the ClusterCollection of one event, otherwise, do not use (loose 1st cl. of 1st event only)
	    first = false;
	    unsigned int  detID0= 0;
	    output.put(outputRange,detID0); // !!! put into adress 0 for detID which will not be used never
	  } //if ( first ) 
	  
	  // put !!!
	  output.put(outputRange,iu);
	  
	  //	} // if(collecto
	
      }   // for
    }   // for
  }   // for
  
  
  // END
  





#ifdef mydigidebug0
  //     check of access to the collector:
  for (int sector=1; sector<sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=1; zside<3; zside++) {
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	collector.clear();
	DigiCollectionFP420::Range outputRange;
	//	outputRange = output->get(iu);
	outputRange = output.get(iu);
	
	// fill output in collector vector (for may be sorting? or other checks)
	std::vector<HDigiFP420> collector;
	//  collector.clear();
	DigiCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	DigiCollectionFP420::ContainerIterator sort_end = outputRange.second;
	for ( ;sort_begin != sort_end; ++sort_begin ) {
	  collector.push_back(*sort_begin);
	} // for
	//  std::sort(collector.begin(),collector.end());
	std::cout <<" ===" << std::endl;
	std::cout <<" ===" << std::endl;
	std::cout <<"=======DigitizerFP420:check of re-new collector size = " << collector.size() << std::endl;
	std::cout <<" iu = " << iu << std::endl;
	std::cout <<" ===" << std::endl;
	vector<HDigiFP420>::const_iterator simHitIter = collector.begin();
	vector<HDigiFP420>::const_iterator simHitIterEnd = collector.end();
	for (;simHitIter != simHitIterEnd; ++simHitIter) {
	  const HDigiFP420 istrip = *simHitIter;
	  
	  std::cout << "DigitizerFP420:check: HDigiFP420:: zside = " << zside << std::endl;
	  std::cout << " strip number=" << istrip.strip() << "  adc=" << istrip.adc() << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" =======================" << std::endl;
	}
	
	//==================================
	
      }   // for
    }   // for
  }   // for
  
  //     end of check of access to the strip collection
  
#endif
  
  //     
  
  
  // Step D: write output to file
  //    iEvent.put(output);
}

//}
//define this as a plug-in

