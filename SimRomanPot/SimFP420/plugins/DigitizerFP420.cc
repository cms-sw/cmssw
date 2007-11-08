///////////////////////////////////////////////////////////////////////////////
// File: DigitizerFP420.cc
// Date: 12.2006
// Description: DigitizerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
//
// system include files
#include <memory>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
///////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/DigitizerFP420.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"

//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"


//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

// G4 stuff
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"
#include "G4TransportationManager.hh"
#include "G4ProcessManager.hh"

#include <cstdlib> 
#include <vector>

using namespace std;

//#include <iostream>




namespace cms
{
DigitizerFP420::DigitizerFP420(const edm::ParameterSet& conf):conf_(conf),stripDigitizer_(new FP420DigiMain(conf)) {


    std::string alias ( conf.getParameter<std::string>("@module_label") );

    produces<DigiCollectionFP420>().setBranchAlias( alias );

    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");

    verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
    sn0   = conf_.getParameter<int>("NumberFP420Stations");
    pn0 = conf_.getParameter<int>("NumberFP420SPlanes");


    if(verbosity>0) {
      std::cout << "Creating a DigitizerFP420" << std::endl;
      std::cout << "DigitizerFP420: sn0=" << sn0 << " pn0=" << pn0 << std::endl;
      std::cout << "DigitizerFP420:trackerContainers.size()=" << trackerContainers.size() << std::endl;

    }
}

// Virtual destructor needed.
  DigitizerFP420::~DigitizerFP420() { 
    if(verbosity>0) {
      std::cout << "Destroying a DigitizerFP420" << std::endl;
    }
    delete stripDigitizer_;
    
  }  
  


  void DigitizerFP420::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)  {
  // be lazy and include the appropriate namespaces
   using namespace edm; 
   using namespace std;   
    
  // Get input
    
    // Step A: Get Inputs for allTrackerHits
    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      iEvent.getByLabel("mix",trackerContainers[i],cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());   }
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));


      std::auto_ptr<DigiCollectionFP420> output(new DigiCollectionFP420);

	SimHitMap.clear();

  // ==================================
  
	MixCollection<PSimHit>::iterator isim;
	for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
	  unsigned int unitID = (*isim).detUnitId();
	  int det, zside, sector, zmodule, sScale = 2*(pn0-1);
	  FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
	  int zScale=2;  unsigned int intindex = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
    if(verbosity>0) {
	  double  losenergy = (*isim).energyLoss();
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ============== DigitizerFP420:  losenergy= " << losenergy << std::endl;
	      std::cout <<" ===" << std::endl;
    }

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
	      output->putclear(inputRange,detID);
	    }//for
	  }//for
	}//for
	//                                                                                                                      !!!!!!   
	// if we want to keep Digi container/Collection for one event uncomment the line below and vice versa
	output->clear();   //container_.clear() --> start from the beginning of the container
	
	//============================================================================================================================================
	
	
	bool first = true;
	for (int sector=1; sector<sn0; sector++) {
	  for (int zmodule=1; zmodule<pn0; zmodule++) {
	    for (int zside=1; zside<3; zside++) {
	      
	      int sScale = 2*(pn0-1);
	      //      int index = CaloNumberingPacker::packCastorIndex(det,zside, sector, zmodule);
	      
	      // intindex is a continues numbering of FP420
	      int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	      // int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
    if(verbosity>0) {
	      int det= 1;
	      int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	      std::cout << " DigitizerFP420:2 run loop   index = " <<  index <<  " iu = " << iu  << std::endl;
    }
	      
	      //    GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
	      //      CLHEP::Hep3Vector Bfieldloc=bfield();
	      G4ThreeVector bfield(  0.,  0.,  0.0  );
	      
    if(verbosity>0) {
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ============== DigitizerFP420:  call run for iu= " << iu << std::endl;
	      std::cout <<" ===" << std::endl;
    }
	      collector.clear();
	      
	      collector= stripDigitizer_->run( SimHitMap[iu],
					       bfield,
					       iu,
					       sScale
					       ); // stripDigitizer_.run...  return 
	      
	      
	      
    if(verbosity>0) {
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::cout <<"=======  DigitizerFP420:  collector size = " << collector.size() << std::endl;
	      std::cout <<" ===" << std::endl;
	      std::cout <<" ===" << std::endl;
    }	      
	      
	      
	      //	if (collector.size()>0){
    if(verbosity>0) {
	      std::cout <<"         ============= DigitizerFP420:collector start!!!!!!!!!!!!!!" << std::endl;
    }
	      DigiCollectionFP420::Range outputRange;
	      outputRange.first = collector.begin();
	      outputRange.second = collector.end();
	      
	      if ( first ) {
		// use it only if ClusterCollectionFP420 is the ClusterCollection of one event, otherwise, do not use (loose 1st cl. of 1st event only)
		first = false;
		unsigned int  detID0= 0;
		output->put(outputRange,detID0); // !!! put into adress 0 for detID which will not be used never
	      } //if ( first ) 
	      
	      // put !!!
	      output->put(outputRange,iu);
	      
	      //	} // if(collecto
	      
	    }   // for
	  }   // for
	}   // for
	
	
	// END
	
	
	
	
	
	
    if(verbosity>0) {
	//     check of access to the collector:
	for (int sector=1; sector<sn0; sector++) {
	  for (int zmodule=1; zmodule<pn0; zmodule++) {
	    for (int zside=1; zside<3; zside++) {
	      int sScale = 2*(pn0-1);
	      int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	      collector.clear();
	      DigiCollectionFP420::Range outputRange;
	      //	outputRange = output->get(iu);
	      outputRange = output->get(iu);
	      
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
	
    }	
	//     
	
	
	
    if(verbosity>0) {
	std::cout << "DigitizerFP420 recoutput" << std::endl;
    }

	iEvent.put(output);
	
  }//produce
  
} // namespace cms

//}
//define this as a plug-in

//DEFINE_FWK_MODULE(DigitizerFP420);
