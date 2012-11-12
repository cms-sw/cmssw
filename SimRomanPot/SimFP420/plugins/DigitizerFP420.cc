/////////////////////////////////////////////////////////////////////////////
// File: DigitizerFP420.cc
// Date: 12.2006
// Description: DigitizerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
//
// system include files
#include <memory>

#include "DataFormats/Common/interface/DetSetVector.h"
//#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
//#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
//#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

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

//#include "SimRomanPot/SimFP420/interface/SimRPUtil.h"
//#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
//#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"

//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
//#include "Geometry/CommonTopologies/interface/StripTopology.h"
//#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//needed for the magnetic field:
//#include "MagneticField/Engine/interface/MagneticField.h"
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
//#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

//Data Base infromations
//#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
//#include "CalibTracker/Records/interface/SiStripGainRcd.h"
//#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
//#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
//#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
//#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
//#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
//#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
//#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
//#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

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
    
    //  produces<edm::DetSetVector<HDigiFP420> >().setBranchAlias( alias );
    //  produces<edm::DetSetVector<HDigiFP420SimLink> >().setBranchAlias ( alias + "hDigiFP420SimLink");
    produces<DigiCollectionFP420>().setBranchAlias( alias );
    
    trackerContainers.clear();
    trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");
    
    verbosity = conf_.getUntrackedParameter<int>("VerbosityLevel");
    dn0   = conf_.getParameter<int>("NumberFP420Detectors");
    sn0   = conf_.getParameter<int>("NumberFP420Stations");
    pn0 = conf_.getParameter<int>("NumberFP420SPlanes");
      rn0 = 7;
    //rn0 = 3;
      theFP420NumberingScheme = new FP420NumberingScheme();
    
    //  produces<DigiCollectionFP420>();
    
    //  produces<StripDigiCollection>();
    //   produces<HDigiFP420>();
    //  produces<edm::DetSetVector<HDigiFP420> >().setBranchAlias( alias );  
    
    //  produces<DigiCollectionFP420>();
    // produces<DigiCollectionFP420>("HDigiFP420");
    
    //  produces<edm::DigiCollectionFP420>();
    
    //   produces<edm::DetSetVector<DigiCollectionFP420> >();
    
    if(verbosity>0) {
      std::cout << "Creating a DigitizerFP420" << std::endl;
      std::cout << "DigitizerFP420: dn0=" << dn0 << " sn0=" << sn0 << " pn0=" << pn0 <<  " rn0=" << rn0 << std::endl;
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
  
  
  
  //  void DigitizerFP420::produce(PSimHitCollection *   theCAFI, DigiCollectionFP420 & output) {
  void DigitizerFP420::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)  {
    // be lazy and include the appropriate namespaces
    using namespace edm; 
    using namespace std;   
    
    if(verbosity>0) {
      std::cout <<" ===" << std::endl;
      std::cout <<" ============== DigitizerFP420: start   produce= " << std::endl;
      std::cout <<" ===" << std::endl;
    }
    // Get input
    //    std::cout << "DigitizerFP420 start produce" << std::endl;
    //  edm::ESHandle < ParticleDataTable > pdt;
    //  iSetup.getData( pdt );
    
    // Step A: Get Inputs for allTrackerHits
    
    
    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      iEvent.getByLabel("mix",trackerContainers[i],cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());   }
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    
    // use instead of the previous
    /*    
	  std::cout <<" ============== DigitizerFP420: start loop           1   " << std::endl;
	  edm::Handle<CrossingFrame<PSimHit> > xFrame;
	  std::cout <<" ============== DigitizerFP420: start loop           2   " << std::endl;
	  iEvent.getByLabel("mix","FP420SI",xFrame);
	  std::cout <<" ============== DigitizerFP420: start loop           3   " << std::endl;
	  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits( new MixCollection<PSimHit>(xFrame.product()) );
	  std::cout <<" ============== DigitizerFP420: start loop           4   " << std::endl;
    */
    
    // use instead of the previous
    /*           
		 edm::Handle<CrossingFrame<PSimHit> > crossingFrame;
		 const std::string FP420HitsName("FP420SI");
		 bool isHit = true;
		 iEvent.getByLabel("mix",FP420HitsName,crossingFrame);
		 MixCollection<PSimHit> * FP420Hits = 0 ;
		 std::cout <<" ============== DigitizerFP420: start loop           1   " << std::endl;
		 //    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(crossingFrame.product()));
		 FP420Hits = new MixCollection<PSimHit>(crossingFrame.product());
		 std::cout <<" ============== DigitizerFP420: start loop           2   " << std::endl;
		 //  if ( ! FP420Hits->inRegistry()  ) isHit = false;
		 //  if ( isHit ) {
		 std::auto_ptr<MixCollection<PSimHit> >  allTrackerHits( FP420Hits );
		 std::cout <<" ============== DigitizerFP420: start loop           3   " << std::endl;
		 //  }  
		 */
    
    //    std::cout << "DigitizerFP420 Step A done" << std::endl;
    
    //Loop on PSimHit
    
    
    ///////////////////////////////////////////////////////////////////////
      // Step C: create empty output collection
      std::auto_ptr<DigiCollectionFP420> output(new DigiCollectionFP420);
      //  std::auto_ptr<edm::DetSetVector<HDigiFP420> > outputfinal(new edm::DetSetVector<HDigiFP420>(output) );
      //  std::auto_ptr<edm::DetSetVector<HDigiFP420> > outputfinal(new edm::DetSetVector<HDigiFP420>(output) );
      //  std::auto_ptr<edm::DetSetVector<HDigiFP420SimLink> > outputlink(new edm::DetSetVector<HDigiFP420SimLink>(output) );
      
      SimHitMap.clear();
      
      // ==================================
      if(verbosity>0) {
	std::cout <<" ===" << std::endl;
	std::cout <<" ============== DigitizerFP420: MixCollection treatment= " << std::endl;
	std::cout <<" ===" << std::endl;
      }
      
      MixCollection<PSimHit>::iterator isim;
      for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
	unsigned int unitID = (*isim).detUnitId();
	int det, zside, sector, zmodule; 
	FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
	// below, the continues plane index should be (for even different sensor index zside)
	//	unsigned int intindex = packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
	unsigned int intindex = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
	//	int zScale=(rn0-1), sScale = (rn0-1)*(pn0-1), dScale = (rn0-1)*(pn0-1)*(sn0-1);
	//	unsigned int intindex = dScale*(det - 1)+sScale*(sector - 1)+zScale*(zmodule - 1)+zside;

	if(verbosity>0) {
	  double  losenergy = (*isim).energyLoss();
	  std::cout <<" ===" << std::endl;
	  std::cout <<" ============== DigitizerFP420:  losenergy= " << losenergy << std::endl;
	  std::cout <<" === for intindex = " << intindex << std::endl;
	}
	// does not matter which index is used: intindex or unitID - mainly to collect hits under every index	
	SimHitMap[intindex].push_back((*isim));
	// for development later one( cal be used another index):
	//	SimHitMap[unitID].push_back((*isim));
      }
      //============================================================================================================================
      
      if(verbosity>0) {
	std::cout <<" ===" << std::endl;
	std::cout <<" ============== DigitizerFP420: put zero to container " << std::endl;
	std::cout <<" ===" << std::endl;
      }
      //    put zero to container info from the beginning (important! because not any detID is updated with coming of new event     !!!!!!   
      // clean info of container from previous event
      for (int det=1; det<dn0; det++) {
	for (int sector=1; sector<sn0; sector++) {
	  for (int zmodule=1; zmodule<pn0; zmodule++) {
	    for (int zside=1; zside<rn0; zside++) {
	      // intindex is a continues numbering of FP420
	      unsigned int detID = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
	      // int zScale=(rn0-1), sScale = (rn0-1)*(pn0-1), dScale = (rn0-1)*(pn0-1)*(sn0-1);
	      // unsigned int detID = dScale*(det - 1)+sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	      std::vector<HDigiFP420> collector;
	      collector.clear();
	      DigiCollectionFP420::Range inputRange;
	      inputRange.first = collector.begin();
	      inputRange.second = collector.end();
	      output->putclear(inputRange,detID);
	    }//for
	  }//for
	}//for
      }//for
      //                                                                                                                      !!!!!!   
      // if we want to keep Digi container/Collection for one event uncomment the line below and vice versa
      output->clear();   //container_.clear() --> start from the beginning of the container
      
      //============================================================================================================================================
      
      
      if(verbosity>0) {
	std::cout <<" ===" << std::endl;
	std::cout <<" ============== DigitizerFP420: start loop over det iu " << std::endl;
	std::cout <<" ============== DigitizerFP420: SimHitMap.size()= " << SimHitMap.size() << std::endl;
	std::cout <<" ===" << std::endl;
      }
      bool first = true;

      /////for development later one
      /*      
      if(verbosity>0) std::cout <<"=======  DigitizerFP420:  SimHitMap size = " << SimHitMap.size() << std::endl;
      for(unsigned int i = 0; i < SimHitMap.size(); i++ ) {
	//	  std::cout <<" ====== DigitizerFP420:                                                      i= " << i << std::endl;
	vector<PSimHit>::const_iterator simHitIter = SimHitMap[i].begin();
	vector<PSimHit>::const_iterator simHitIterEnd = SimHitMap[i].end();
	for (;simHitIter != simHitIterEnd; ++simHitIter) {
	  const PSimHit ihit = *simHitIter;
	  unsigned int unitID = ihit.detUnitId();
	  if(verbosity>0) std::cout <<" ====== DigitizerFP420: unitID= " << unitID << " i=  " << i << std::endl;
	  int det, zside, sector, zmodule; 
	  FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
	  int zScale=(rn0-1), sScale = (rn0-1)*(pn0-1), dScale = (rn0-1)*(pn0-1)*(sn0-1);
	  unsigned int iu = dScale*(det - 1)+sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	}
      }
*/
      ////////////////////////
      //============================================================================================================================================
      // new:   <------
      for(unsigned int i = 0; i < SimHitMap.size(); i++ ) {
	vector<PSimHit>::const_iterator simHitIter = SimHitMap[i].begin();
	vector<PSimHit>::const_iterator simHitIterEnd = SimHitMap[i].end();
	for (;simHitIter != simHitIterEnd; ++simHitIter) {
	  const PSimHit ihit = *simHitIter;
	  unsigned int unitID = ihit.detUnitId();
	  if(verbosity>0 || verbosity==-50) std::cout <<" ====== DigitizerFP420: unitID= " << unitID << "Hit number i=  " << i << std::endl;
	  int det, zside, sector, zmodule; 
	  FP420NumberingScheme::unpackFP420Index(unitID, det, zside, sector, zmodule);
	  // <------
	  // old: <------
	  //     for (int det=1; det<dn0; det++) {
	  //	for (int sector=1; sector<sn0; sector++) {
	  // for (int zmodule=1; zmodule<pn0; zmodule++) {
	  //  for (int zside=1; zside<rn0; zside++) {
	  // <------
	  
	  
	  
	  
	  
	  unsigned int iu = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
	  if(verbosity>0 || verbosity==-50) std::cout <<"for Hits iu = " << iu <<" sector = " << sector <<" zmodule = " << zmodule <<" zside = " << zside << "  det=" << det << std::endl;
	  //   int zScale=(rn0-1), sScale = (rn0-1)*(pn0-1), dScale = (rn0-1)*(pn0-1)*(sn0-1);
	  //  unsigned int iu = dScale*(det - 1)+sScale*(sector - 1)+zScale*(zmodule - 1)+zside;

	  if(verbosity>0) {
	    unsigned int index = theFP420NumberingScheme->FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);
	    std::cout << " DigitizerFP420:       index = " <<  index <<  " iu = " << iu  << std::endl;
	  }
	  
	  //    GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
	  //      CLHEP::Hep3Vector Bfieldloc=bfield();
	  G4ThreeVector bfield(  0.,  0.,  0.0  );
	  //	G4ThreeVector bfield(  0.5,  0.5,  1.0  );
	  
	  
	  if(verbosity>0) {
	    std::cout <<" ===" << std::endl;
	    std::cout <<" ============== DigitizerFP420:  call run for iu= " << iu << std::endl;
	    std::cout <<" ===" << std::endl;
	  }
	  collector.clear();
	  
	  collector= stripDigitizer_->run( SimHitMap[iu],
					   bfield,
					   iu
					   ); // stripDigitizer_.run...  return 
	  //						 ,sScale
	  
	  
	  
	  if(verbosity>0) {
	    std::cout <<" ===" << std::endl;
	    std::cout <<" ===" << std::endl;
	    std::cout <<"=======  DigitizerFP420:  collector size = " << collector.size() << std::endl;
	    std::cout <<" ===" << std::endl;
	    std::cout <<" ===" << std::endl;
	  }	      
	  /*		
	    
	  std::vector<HDigiFP420> collector;
	  collector.clear();
	  DigiCollectionFP420::Range inputRange;
	  inputRange.first = collector.begin();
	  inputRange.second = collector.end();
	  output->putclear(inputRange,detID);
	  */
	  if (collector.size()>0){
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
	    
	  } // if(collector.size()>0
	  
	  //   }   // for
	  // }   // for
	  //	}   // for
	  // }   // for
	  
	}//for
      }//for
      
      // END
      
      /*      
      if(verbosity>0) {
	std::vector<HDigiFP420> theAllDigis;
	theAllDigis.clear();
	DigiCollectionFP420::Range outputRange;
	DigiCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	DigiCollectionFP420::ContainerIterator sort_end = outputRange.second;
	theAllDigis.insert(theAllDigis.end(), sort_begin, sort_end);
	std::cout <<"======  theAllDigis size = " << theAllDigis.size() << std::endl;
	for (std::vector<HDigiFP420>::iterator isim = theAllDigis.begin();
	     isim != theAllDigis.end(); ++isim){
	  const HDigiFP420 istrip = *isim;
	  std::cout << "*******************************************DigitizerFP420:check1" << std::endl;
	  std::cout << " strip number=" << istrip.strip() << "  adc=" << istrip.adc() << std::endl;
	  std::cout <<" channel =" << istrip.channel() <<" V " << istrip.stripV() <<" VW " << istrip.stripVW() << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" ===" << std::endl;
	  std::cout <<" =======================" << std::endl;
	}// for
      }
*/
      if(verbosity==-50) {
	//     check of access to the collector:
	for (int det=1; det<dn0; det++) {
	  for (int sector=1; sector<sn0; sector++) {
	    for (int zmodule=1; zmodule<pn0; zmodule++) {
	      for (int zside=1; zside<rn0; zside++) {
		unsigned int iu = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0, pn0, sn0, det, zside, sector, zmodule);
		int layer = theFP420NumberingScheme->FP420NumberingScheme::unpackLayerIndex(rn0,zside);
		int orient = theFP420NumberingScheme->FP420NumberingScheme::unpackOrientation(rn0,zside);
		std::cout << "****DigitizerFP420:check2" << std::endl;
		//	std::cout <<" iu = " << iu <<" sector = " << sector <<" zmodule = " << zmodule <<" zside = " << zside << "  det=" << det << std::endl;
		//	std::cout <<" layer = " << layer <<" orient = " << orient << std::endl;
		int newdet, newzside, newsector, newzmodule;
		theFP420NumberingScheme->FP420NumberingScheme::unpackMYIndex(iu, rn0, pn0, sn0, newdet, newzside, newsector, newzmodule);
		std::cout <<" newdet = " << newdet <<" newsector = " << newsector <<" newzmodule = " << newzmodule <<" newzside = " << newzside << std::endl;
		
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
		std::cout <<"======  collector size = " << collector.size() << std::endl;
		if(collector.size()>0) {
		  std::cout <<" iu = " << iu <<" sector = " << sector <<" zmodule = " << zmodule <<" zside = " << zside << "  det=" << det <<" layer = " << layer <<" orient = " << orient << std::endl;
		  std::cout <<" ===" << std::endl;
		}
		vector<HDigiFP420>::const_iterator simHitIter = collector.begin();
		vector<HDigiFP420>::const_iterator simHitIterEnd = collector.end();
		for (;simHitIter != simHitIterEnd; ++simHitIter) {
		  const HDigiFP420 istrip = *simHitIter;
		  std::cout << " strip number=" << istrip.strip() << "  adc=" << istrip.adc() << std::endl;
		  std::cout <<" channel =" << istrip.channel() <<" V " << istrip.stripV() <<" VW " << istrip.stripVW() << std::endl;
		  std::cout <<" ===" << std::endl;
		  std::cout <<" ===" << std::endl;
		  std::cout <<" ===================================================" << std::endl;
		}
		
		//==================================
		
	      }   // for
	    }   // for
	  }   // for
	}   // for
	
	//     end of check of access to the strip collection
	
      }// if(verbosity	
      //     
      
      
      // Step D: write output to file
      //    iEvent.put(output);
      
      if(verbosity>0) {
	std::cout << "DigitizerFP420 recoutput" << std::endl;
      }
      // Step D: write output to file
      iEvent.put(output);
      // iEvent.put(outputlink);
      //	  iEvent.put(pDigis);
      
      // Step D: write output to file 
      //  iEvent.put(output);
      //  iEvent.put(outputlink);
      //-------------------------------------------------------------------
      //    std::cout << "DigitizerFP420 recoutput" << std::endl;
      //	  iEvent.put(pDigis);
      
      
      
      
  }//produce
  
} // namespace cms

//}
//define this as a plug-in

//DEFINE_FWK_MODULE(DigitizerFP420);
