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

typedef float HGCSimData_t;

//15 time samples: 9 pre-samples, 1 in-time, 5 post-samples
typedef std::array<HGCSimData_t,15> HGCSimHitData;

//1st array=energy, 2nd array=energy weighted time-of-flight
typedef std::unordered_map<uint32_t, std::array<HGCSimHitData,2> > HGCSimHitDataAccumulator; 

template <class D>
class HGCDigitizerBase {
 public:
 
  typedef edm::SortedCollection<D> DColl;

  /**
     @short CTOR
   */
 HGCDigitizerBase(const edm::ParameterSet &ps) : simpleNoiseGen_(0)
  {
    bxTime_        = ps.getParameter<double>("bxTime");
    myCfg_         = ps.getParameter<edm::ParameterSet>("digiCfg"); 
    doTimeSamples_ = myCfg_.getParameter< bool >("doTimeSamples");
    mipInKeV_      = myCfg_.getParameter<double>("mipInKeV");
    if(myCfg_.exists("mipInfC")) mipInfC_ = myCfg_.getParameter<double>("mipInfC");
    else                         mipInfC_ = 0;
    mip2noise_     = myCfg_.getParameter<double>("mip2noise");

    edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
    myFEelectronics_ = std::unique_ptr<HGCFEElectronics<D> >( new HGCFEElectronics<D>(feCfg) );
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
    if(digitizationType==0) runSimple(digiColl,simData);
    else                    runDigitizer(digiColl,simData,digitizationType);
  }


  /**
     @short a trivial digitization: sum energies and digitize without noise
   */
  void runSimple(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData)
  {
    for(HGCSimHitDataAccumulator::iterator it=simData.begin();
	it!=simData.end();
	it++)
      {
	std::vector<float> chargeColl( it->second[0].size(), 0 ),toa( it->second[0].size(), 0 );
	for(size_t i=0; i<it->second[0].size(); i++) 
	  {
	    //convert total energy GeV->keV counts
	    double totalEn=(it->second)[0][i]*1e6;
	    
	    //time of arrival
	    if(totalEn>0) toa[i]=(it->second)[1][i]*1e6/totalEn;
	    
	    ////bool debug(totalEn>0);
	    ////if(debug) std::cout << "[" << i << "] " << totalEn << "keV ->";

	    //add noise (in keV)
	    double noiseEn=simpleNoiseGen_->fire();
	    totalEn += noiseEn;
	    if(totalEn<0) totalEn=0;

	    //convert keV -> MIP -> fC
	    chargeColl[i]= ((totalEn/mipInKeV_)* mipInfC_);
	  }
	
	//run the shaper to create a new data frame
	D rawDataFrame( it->first );
	myFEelectronics_->runShaper(rawDataFrame,chargeColl,toa);
	
	//update the output according to the final shape
	updateOutput(coll,rawDataFrame);
      }
  }

  /**
     @short prepares the output according to the number of time samples to produce
   */
  void updateOutput(std::auto_ptr<DColl> &coll,D rawDataFrame)
  {
    int itIdx(rawDataFrame.size()-1);
    if(itIdx<0) return;

    //check if any of the samples is above threshold and save result
    if(doTimeSamples_)
      {
	//bool keep(false);
	//for(int it=0; it<rawDataFrame.size(); it++)	
	//  if(rawDataFrame[it].data() >= adcThreshold_ ) 
	//keep=true;
	//if(keep) coll->push_back(rawDataFrame);
	coll->push_back(rawDataFrame);
      }
    //check if in-time sample is above threshold and put result into the event
    else
      {
	//create a new data frame containing only the in-time digi
	D singleRawDataFrame( rawDataFrame.id() );
	singleRawDataFrame.resize(1);

	HGCSample singleSample;
	singleSample.set(rawDataFrame[itIdx].threshold(),
			 rawDataFrame[itIdx].mode(),
			 rawDataFrame[itIdx].toa(),
			 rawDataFrame[itIdx].data());
	singleRawDataFrame.setSample(0, singleSample);
	//if(singleRawDataFrame[0].data() < adcThreshold_ ) return;
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

  //a simple noise generator
  mutable CLHEP::RandGauss *simpleNoiseGen_;
  
  //parameters for the trivial digitization scheme
  double mipInKeV_, mipInfC_,  mip2noise_;
  
  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<D> > myFEelectronics_;

  //bunch time
  double bxTime_;
  
  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

 
 private:

};

#endif
