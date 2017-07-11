#ifndef SimCalorimetry_HGCSimProducers_hgcdigitizerbase
#define SimCalorimetry_HGCSimProducers_hgcdigitizerbase

#include <array>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerTypes.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCFEElectronics.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

namespace hgc = hgc_digi;

template <class DFr>
class HGCDigitizerBase {
 public:
  
  typedef DFr DigiType;

  typedef edm::SortedCollection<DFr> DColl;
  
  /**
     @short CTOR
  */
  HGCDigitizerBase(const edm::ParameterSet &ps); 
 /**
    @short steer digitization mode
 */
  void run(std::unique_ptr<DColl> &digiColl, hgc::HGCSimHitDataAccumulator &simData, 
	   const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
	   uint32_t digitizationType,CLHEP::HepRandomEngine* engine);
  
  /**
     @short getters
  */
  float keV2fC() const { return keV2fC_; }
  bool toaModeByEnergy() const { return (myFEelectronics_->toaMode()==HGCFEElectronics<DFr>::WEIGHTEDBYE); }
  float tdcOnset() const { return myFEelectronics_->getTDCOnset(); }
  std::array<float,3> tdcForToaOnset() const { return myFEelectronics_->getTDCForToaOnset(); }

  /**
     @short a trivial digitization: sum energies and digitize without noise
   */
  void runSimple(std::unique_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData, 
		 const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
		 CLHEP::HepRandomEngine* engine);
  
  /**
     @short prepares the output according to the number of time samples to produce
  */
  void updateOutput(std::unique_ptr<DColl> &coll, const DFr& rawDataFrame);
  
  /**
     @short to be specialized by top class
  */
  virtual void runDigitizer(std::unique_ptr<DColl> &coll, hgc::HGCSimHitDataAccumulator &simData,
			    const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
			    uint32_t digitizerType, CLHEP::HepRandomEngine* engine)
  {
    throw cms::Exception("HGCDigitizerBaseException") << " Failed to find specialization of runDigitizer";
  }
  
  /**
     @short DTOR
  */
  virtual ~HGCDigitizerBase() 
    { };
  
  

 protected:
  
  //baseline configuration
  edm::ParameterSet myCfg_;
  
  //1keV in fC
  float keV2fC_;
  
  //noise level
  std::vector<float> noise_fC_;

  //charge collection efficiency
  std::vector<double> cce_;
  
  //front-end electronics model
  std::unique_ptr<HGCFEElectronics<DFr> > myFEelectronics_;

  //bunch time
  double bxTime_;
  
  //if true will put both in time and out-of-time samples in the event
  bool doTimeSamples_;

};

#endif
