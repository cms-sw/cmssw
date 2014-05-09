#ifndef SimCalorimetry_HGCSimProducers_hgcdigitizerbase
#define SimCalorimetry_HGCSimProducers_hgcdigitizerbase

#include <iostream>
#include <vector>
#include <memory>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/EDMException.h"

typedef std::vector<double> HGCSimHitData;
typedef std::map<uint32_t, HGCSimHitData> HGCSimHitDataAccumulator;

template <class D>
class HGCDigitizerBase {
 public:
  
  typedef edm::SortedCollection<D> DColl;

  /**
     @short CTOR
   */
  HGCDigitizerBase() {};

  /**
     @short steer digitization mode
   */
  void run(std::auto_ptr<DColl> &digiColl,HGCSimHitDataAccumulator &simData,bool doTrivialDigis)
  {
    if(doTrivialDigis) runTrivial(digiColl,simData);
    else               runDigitizer(digiColl,simData);
  }


  /**
     @short a trivial digitization: sum energies and digitize without noise
   */
  void runTrivial(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData)
  {
    for(HGCSimHitDataAccumulator::iterator it=simData.begin();
	it!=simData.end();
	it++)
      {

	//convert total energy GeV->10 keV=LSB
	double totalEn(0);
	for(size_t i=0; i<it->second.size(); i++) totalEn+= (it->second)[i];
	totalEn*=1e5;
	HGCSample singleSample( (uint16_t) totalEn );
	if(singleSample.adc()==0) continue;
 
	//no time information
	D newDataFrame( it->first );
	newDataFrame.setSample(0, singleSample);

	//add to collection to produce
	coll->push_back(newDataFrame);
      }
  }

  /**
     @short to be specialized by top class
   */
  virtual void runDigitizer(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData)
  {
    throw cms::Exception("HGCDigitizerBaseException") << " Failed to find specialization of runDigitizer";
  }

  /**
     @short DTOR
   */
  ~HGCDigitizerBase() {};
  
 private:

};

#endif
