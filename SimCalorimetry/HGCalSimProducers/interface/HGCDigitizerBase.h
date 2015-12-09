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

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerTypes.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"

namespace hgc = hgc_digi;

template <class DFr>
class HGCDigitizerBase {
 public:
  
  typedef edm::SortedCollection<DFr> DColl;
  
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
    myFEelectronics_        = std::unique_ptr<HGCFEElectronics<DFr> >( new HGCFEElectronics<DFr>(feCfg) );
  }
      
 /**
    @short steer digitization mode
 */
  void run(std::auto_ptr<DColl> &digiColl, hgc::HGCSimHitDataAccumulator &simData, uint32_t digitizationType,CLHEP::HepRandomEngine* engine);
  
  /**
     @short getters
  */
  float keV2fC() const { return keV2fC_; }
  bool toaModeByEnergy() const { return (myFEelectronics_->toaMode()==HGCFEElectronics<DFr>::WEIGHTEDBYE); }
  float tdcOnset() const { return myFEelectronics_->getTDCOnset(); }

  /**
     @short a trivial digitization: sum energies and digitize without noise
   */
  void runSimple(std::auto_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine);
  
  /**
     @short prepares the output according to the number of time samples to produce
  */
  void updateOutput(std::auto_ptr<DColl> &coll, const DFr& rawDataFrame);
  
  /**
     @short to be specialized by top class
  */
  virtual void runDigitizer(std::auto_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData,uint32_t digitizerType, CLHEP::HepRandomEngine* engine)
  {
    throw cms::Exception("HGCDigitizerBaseException") << " Failed to find specialization of runDigitizer";
  }
  
  /**
     @short DTOR
  */
  ~HGCDigitizerBase() 
    { };
  
  

 protected:
  
  //baseline configuration
  edm::ParameterSet myCfg_;
  
  //1keV in fC
  float keV2fC_;
  
  //noise level
  float noise_fC_;
  
  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<DFr> > myFEelectronics_;

  //bunch time
  double bxTime_;
  
  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

};

#endif
