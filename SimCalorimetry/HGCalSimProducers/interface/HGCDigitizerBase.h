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
#include "CLHEP/Random/RandGaussQ.h"

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
 HGCDigitizerBase(const edm::ParameterSet &ps) {
    myCfg_         = ps.getParameter<edm::ParameterSet>("digiCfg"); 
    bxTime_        = ps.getParameter<int32_t>("bxTime");
    doTimeSamples_ = myCfg_.getParameter< bool >("doTimeSamples");
    mipInKeV_      = myCfg_.getParameter<double>("mipInKeV");
    lsbInMIP_      = myCfg_.getParameter<double>("lsbInMIP");
    mip2noise_     = myCfg_.getParameter<double>("mip2noise");
    adcThreshold_  = myCfg_.getParameter< uint32_t >("adcThreshold");
    shaperN_       = myCfg_.getParameter< double >("shaperN");
    shaperTau_     = myCfg_.getParameter< double >("shaperTau");
  }
  
  /**
     @short steer digitization mode
   */
  void run(std::auto_ptr<DColl> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine) {
    if(digitizationType==0) runTrivial(digiColl,simData,engine);
    else                    runDigitizer(digiColl,simData,digitizationType,engine);
  }


  /**
     @short a trivial digitization: sum energies and digitize without noise
  */
  void runTrivial(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine) {
    for(HGCSimHitDataAccumulator::iterator it=simData.begin();
	it!=simData.end(); it++) {
      
      //create a new data frame
      D rawDataFrame( it->first );

      for(size_t i=0; i<it->second.size(); i++) {
	//convert total energy GeV->keV->ADC counts
	double totalEn=(it->second)[i]*1e6;

	//add noise (in keV)
	double noiseEn=CLHEP::RandGaussQ::shoot(engine,0,mipInKeV_/mip2noise_);
	totalEn += noiseEn;
	if(totalEn<0) totalEn=0;
	
	//round to integer (sample will saturate the value according to available bits)
	uint16_t totalEnInt = floor( (totalEn/mipInKeV_) / lsbInMIP_ );

	//0 gain for the moment
	HGCSample singleSample;
	singleSample.set(0, totalEnInt);
	
	rawDataFrame.setSample(i, singleSample);
      }
	
      //run the shaper
      runShaper(rawDataFrame);

      //update the output according to the final shape
      updateOutput(coll,rawDataFrame);
    }
  }

  /**
     @short prepares the output according to the number of time samples to produce
  */
  void updateOutput(std::auto_ptr<DColl> &coll,D rawDataFrame) {
    size_t itIdx(4); //index to the in-time digi - this could be configurable in a future version

    //check if in-time sample is above threshold and put result into the event
    if(doTimeSamples_) {
      if(rawDataFrame[itIdx].adc() < adcThreshold_ ) return;
      coll->push_back(rawDataFrame);
    } else {
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
     @short applies a shape to each time sample and propagates the tails to the subsequent time samples
   */
  void runShaper(D &dataFrame) {
    std::vector<uint16_t> oldADC(dataFrame.size());
    for(int it=0; it<dataFrame.size(); it++) {
      uint16_t gain=dataFrame[it].gain();
      oldADC[it]=dataFrame[it].adc();
      uint16_t newADC(oldADC[it]);
	
      if(shaperN_*shaperTau_>0){
	for(int jt=0; jt<it; jt++) {
	  float relTime(bxTime_*(it-jt)+shaperN_*shaperTau_);	
	  newADC += uint16_t(oldADC[jt]*pow(relTime/(shaperN_*shaperTau_),shaperN_)*exp(-(relTime-shaperN_*shaperTau_)/shaperTau_));	      
	}
      }

      HGCSample newSample;
      newSample.set(gain,newADC);
      dataFrame.setSample(it,newSample);
    }
  }

  /**
     @short to be specialized by top class
   */
  virtual void runDigitizer(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData,uint32_t digitizerType, CLHEP::HepRandomEngine* engine) {
    throw cms::Exception("HGCDigitizerBaseException") << " Failed to find specialization of runDigitizer";
  }

  /**
     @short DTOR
  */
  ~HGCDigitizerBase() { };
  
  //baseline configuration
  edm::ParameterSet myCfg_;

  //minimum ADC counts to produce DIGIs
  uint32_t adcThreshold_;
  
  //parameters for the trivial digitization scheme
  double mipInKeV_, lsbInMIP_, mip2noise_;
  
  //parameters for trivial shaping
  double shaperN_, shaperTau_;

  //bunch time
  int bxTime_;
  
  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

private:

};

#endif
