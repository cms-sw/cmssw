#ifndef SimCalorimetry_HGCSimProducers_hgcdigitizerbase
#define SimCalorimetry_HGCSimProducers_hgcdigitizerbase

#include <iostream>
#include <vector>
#include <memory>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Random/RandGauss.h"

typedef std::vector<double> HGCSimHitData;
typedef std::map<uint32_t, HGCSimHitData> HGCSimHitDataAccumulator;

template <class D>
class HGCDigitizerBase {
 public:
 
  typedef edm::SortedCollection<D> DColl;

  /**
     @short CTOR
   */
  HGCDigitizerBase(const edm::ParameterSet &ps) : simpleNoiseGen_(0)
    {
      myCfg_      = ps.getUntrackedParameter<edm::ParameterSet>("digiCfg"); 
      lsbInMeV_   = myCfg_.getUntrackedParameter<double>("lsbInMeV");
      noiseInMeV_ = myCfg_.getUntrackedParameter<double>("noiseInMeV");
    }

  /**
     @short init a random number generator for noise
   */
  void setRandomNumberEngine(CLHEP::HepRandomEngine& engine) 
  { 
    simpleNoiseGen_ = new CLHEP::RandGauss(engine,0,noiseInMeV_);
  }

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

	//convert total energy GeV->ADC counts
	double totalEn(0);
	for(size_t i=0; i<it->second.size(); i++) totalEn+= (it->second)[i];
	totalEn*=1e6;

	//add noise (in MeV)
	double noiseEn=simpleNoiseGen_->fire();
	if(noiseEn<0) noiseEn=0;
 
	//round to integer
	uint16_t totalEnInt = floor((totalEn+noiseEn)/lsbInMeV_);

	HGCSample singleSample( totalEnInt );
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
  ~HGCDigitizerBase() 
    {
      if(simpleNoiseGen_) delete simpleNoiseGen_;
    };
  
  //baseline configuration
  edm::ParameterSet myCfg_;

 private:

  //
  double lsbInMeV_,noiseInMeV_;

  //
  mutable CLHEP::RandGauss *simpleNoiseGen_;

};

#endif
