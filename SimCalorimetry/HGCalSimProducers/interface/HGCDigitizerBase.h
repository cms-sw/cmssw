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

#include "SimCalorimetry/HGCSimProducers/src/HGCFEElectronics.cc"

typedef float HGCSimEn_t;
typedef std::array<HGCSimEn_t,6> HGCSimHitData;
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
    bxTime_        = ps.getParameter<int32_t>("bxTime");
    doTimeSamples_ = myCfg_.getParameter< bool >("doTimeSamples");
    mipInKeV_      = myCfg_.getParameter<double>("mipInKeV");
    mip2noise_     = myCfg_.getParameter<double>("mip2noise");
    adcThreshold_  = myCfg_.getParameter< uint32_t >("adcThreshold");

    edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
    myFEelectronics_ = std::unique_ptr<HGCFEElectronics<D> >( new HGCFEElectronics<D>(feCfg, bxTime_) );
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
	D rawDataFrame( it->first );

	for(size_t i=0; i<it->second.size(); i++) 
	  {
	    //convert total energy GeV->keV->ADC counts
	    double totalEn=(it->second)[i]*1e6;

	    //add noise (in keV)
	    double noiseEn=simpleNoiseGen_->fire();
	    totalEn += noiseEn;
	    if(totalEn<0) totalEn=0;
	
	    //round to integer (sample will saturate the value according to available bits)
	    uint16_t totalEnInt = floor( (totalEn*(myFEelectronics_->getLSB()))/mipInKeV_ );

	    //0 gain for the moment
	    HGCSample singleSample;
	    singleSample.set(0, totalEnInt);
	    
	    rawDataFrame.setSample(i, singleSample);
	  }
	
	//run the shaper
	myFEelectronics_->runShaper(rawDataFrame);

	//update the output according to the final shape
	updateOutput(coll,rawDataFrame);
      }
  }

  /**
     @short prepares the output according to the number of time samples to produce
   */
  void updateOutput(std::auto_ptr<DColl> &coll,D rawDataFrame)
  {
    size_t itIdx(4); //index to the in-time digi - this could be configurable in a future version

    //check if in-time sample is above threshold and put result into the event
    if(doTimeSamples_)
      {
	if(rawDataFrame[itIdx].adc() < adcThreshold_ ) return;
	coll->push_back(rawDataFrame);
      }
    else
      {
	//create a new data frame containing only the in-time digi
	D singleRawDataFrame( rawDataFrame.id() );
	singleRawDataFrame.resize(1);

	HGCSample singleSample;
	singleSample.set(rawDataFrame[itIdx].gain(),rawDataFrame[itIdx].adc());
	singleRawDataFrame.setSample(0, singleSample);
	if(singleRawDataFrame[0].adc() < adcThreshold_ ) return;
	coll->push_back(singleRawDataFrame);
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
  double mipInKeV_,  mip2noise_;
  
  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<D> > myFEelectronics_;

  //bunch time
  int bxTime_;
  
  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

 private:

};

#endif
