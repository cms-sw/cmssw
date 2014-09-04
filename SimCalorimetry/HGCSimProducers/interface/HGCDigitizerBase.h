#ifndef SimCalorimetry_HGCSimProducers_hgcdigitizerbase
#define SimCalorimetry_HGCSimProducers_hgcdigitizerbase

#include <array>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Random/RandGauss.h"

typedef float HGCSimEn_t;
typedef std::array<HGCSimEn_t,3> HGCSimHitData;
typedef std::unordered_map<uint32_t, HGCSimHitData> HGCSimHitDataAccumulator;

template <class D>
class HGCDigitizerBase {
 public:
 
  typedef edm::SortedCollection<D> DColl;

  /**
     @short CTOR
   */
  HGCDigitizerBase(const edm::ParameterSet &ps) : simpleNoiseGen_(0)
    {
      myCfg_         = ps.getParameter<edm::ParameterSet>("digiCfg"); 
      mipInKeV_      = myCfg_.getParameter<double>("mipInKeV");
      lsbInMIP_      = myCfg_.getParameter<double>("lsbInMIP");
      mip2noise_     = myCfg_.getParameter<double>("mip2noise");
      adcThreshold_  = myCfg_.getParameter< uint32_t >("adcThreshold");
    }

  /**
     @short init a random number generator for noise
   */
  void setRandomNumberEngine(CLHEP::HepRandomEngine& engine) 
  {       
    simpleNoiseGen_ = new CLHEP::RandGauss(engine,0,mipInKeV_/mip2noise_ );
  }
  
  /**
     @short steer digitization mode
   */
  void run(std::auto_ptr<DColl> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType)
  {
    if(digitizationType==0) runTrivial(digiColl,simData);
    else                    runDigitizer(digiColl,simData,digitizationType);
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
	
	//create a new data frame
	D newDataFrame( it->first );

	for(size_t i=0; i<it->second.size(); i++) 
	  {

	    //convert total energy GeV->keV->ADC counts
	    double totalEn=(it->second)[i]*1e6;

	    //add noise (in keV)
	    double noiseEn=simpleNoiseGen_->fire();
	    totalEn += noiseEn;
	    if(totalEn<0) totalEn=0;
	
	    //round to integer (sample will saturate the value according to available bits)
	    uint16_t totalEnInt = floor( (totalEn/mipInKeV_) / lsbInMIP_ );

	    //0 gain for the moment
	    HGCSample singleSample;
	    singleSample.set(0, totalEnInt);
	    
	    newDataFrame.setSample(i, singleSample);
	  }
	
	//add to collection to produce
	coll->push_back(newDataFrame);
      }
  }

  /**
     @short to be specialized by top class
   */
  virtual void runDigitizer(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData,uint32_t digitizerType)
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

  //minimum ADC counts to produce DIGIs
  uint32_t adcThreshold_;

  //a simple noise generator
  mutable CLHEP::RandGauss *simpleNoiseGen_;
  
  //parameters for the trivial digitization scheme
  double mipInKeV_, lsbInMIP_, mip2noise_;
  
 private:

};

#endif
