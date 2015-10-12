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
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"

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
 HGCDigitizerBase(const edm::ParameterSet &ps)  {
    bxTime_        = ps.getParameter<double>("bxTime");
    myCfg_         = ps.getParameter<edm::ParameterSet>("digiCfg"); 
    doTimeSamples_ = myCfg_.getParameter< bool >("doTimeSamples");
    if(myCfg_.exists("keV2fC"))   keV2fC_   = myCfg_.getParameter<double>("keV2fC");    
    else                          keV2fC_   = 1.0;
    if(myCfg_.exists("noise_fC")) noise_fC_ = myCfg_.getParameter<double>("noise_fC");
    else                          noise_fC_ = 1.0;
    edm::ParameterSet feCfg = myCfg_.getParameter<edm::ParameterSet>("feCfg");
    myFEelectronics_        = std::unique_ptr<HGCFEElectronics<D> >( new HGCFEElectronics<D>(feCfg) );
  }
      
  /**
     @short steer digitization mode
   */
  void run(std::auto_ptr<DColl> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType,CLHEP::HepRandomEngine* engine)
  {
    if(digitizationType==0) runSimple(digiColl,simData,engine);
    else                    runDigitizer(digiColl,simData,digitizationType,engine);
  }

  /**
     @short getters
   */
  float keV2fC() { return keV2fC_; }
  bool toaModeByEnergy() { return (myFEelectronics_->toaMode()==HGCFEElectronics<D>::WEIGHTEDBYE); }
  float tdcOnset() { return myFEelectronics_->getTDCOnset(); }

  /**
     @short a trivial digitization: sum energies and digitize without noise
   */
  void runSimple(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine)
  {
    CLHEP::RandGauss simpleNoiseGen(engine,0,noise_fC_);
    for(HGCSimHitDataAccumulator::iterator it=simData.begin();
	it!=simData.end();
	it++)
      {
	std::vector<float> chargeColl( it->second[0].size(), 0 ),toa( it->second[0].size(), 0 );
	for(size_t i=0; i<it->second[0].size(); i++) 
	  {
	    double rawCharge((it->second)[0][i]);

	    //time of arrival
	    toa[i]=(it->second)[1][i];
	    if(myFEelectronics_->toaMode()==HGCFEElectronics<D>::WEIGHTEDBYE && rawCharge>0) 
	      toa[i]=(it->second)[1][i]/rawCharge;
	    
	    //convert total energy in GeV to charge (fC)
	    //double totalEn=rawEn*1e6*keV2fC_;
	    double totalCharge=rawCharge;

	    //add noise (in fC)
	    //we assume it's randomly distributed and won't impact ToA measurement
	    totalCharge += std::max(simpleNoiseGen.fire(),0.);
	    if(totalCharge<0) totalCharge=0;

	    chargeColl[i]= totalCharge;
	  }
	
	//run the shaper to create a new data frame
	D rawDataFrame( it->first );
	myFEelectronics_->runShaper(rawDataFrame,chargeColl,toa,engine);
	
	//update the output according to the final shape
	updateOutput(coll,rawDataFrame);
      }
  }

  /**
     @short prepares the output according to the number of time samples to produce
   */
  void updateOutput(std::auto_ptr<DColl> &coll,D rawDataFrame)
  {
    int itIdx(9);
    if(rawDataFrame.size()<=itIdx+2) return;
    
    D dataFrame( rawDataFrame.id() );
    dataFrame.resize(5);
    bool putInEvent(false);
    for(int it=0;it<5; it++) 
      {
	HGCSample singleSample;
	singleSample.set(rawDataFrame[itIdx-2+it].threshold(),
			 rawDataFrame[itIdx-2+it].mode(),
			 rawDataFrame[itIdx-2+it].toa(),
			 rawDataFrame[itIdx-2+it].data());
	dataFrame.setSample(it, singleSample);
	if(it==2) { putInEvent=rawDataFrame[itIdx-2+it].threshold(); }
      }
    if(putInEvent) coll->push_back(dataFrame);
  }

  /**
     @short to be specialized by top class
   */
  virtual void runDigitizer(std::auto_ptr<DColl> &coll,HGCSimHitDataAccumulator &simData,uint32_t digitizerType, CLHEP::HepRandomEngine* engine)
  {
    throw cms::Exception("HGCDigitizerBaseException") << " Failed to find specialization of runDigitizer";
  }

  /**
     @short DTOR
   */
  ~HGCDigitizerBase() 
    { };
  
  //baseline configuration
  edm::ParameterSet myCfg_;
  
  //1keV in fC
  float keV2fC_;
  
  //noise level
  float noise_fC_;
  
  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<D> > myFEelectronics_;

  //bunch time
  double bxTime_;
  
  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

 
 private:

};

#endif
