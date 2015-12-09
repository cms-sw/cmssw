#ifndef HGCalSimProducers_HGCDigitizer_h
#define HGCalSimProducers_HGCDigitizer_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimCalorimetry/HGCalSimProducers/interface/HGCEEDigitizer.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEfrontDigitizer.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCHEbackDigitizer.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include <vector>
#include <map>
#include <memory>
#include <tuple>

class PCaloHit;
class PileUpEventPrincipal;

class HGCDigitizer
{
public:
  
  HGCDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  ~HGCDigitizer() { }

  typedef std::tuple<int,uint32_t,float> HGCCaloHitTuple_t;
  static bool orderByDetIdThenTime(const HGCCaloHitTuple_t &a, const HGCCaloHitTuple_t &b)
  {
    unsigned int detId_a(std::get<1>(a)), detId_b(std::get<1>(b));

    if(detId_a<detId_b) return true;
    if(detId_a>detId_b) return false;

    double time_a(std::get<2>(a)), time_b(std::get<2>(b));
    if(time_a<time_b) return true;

    return false;
  }


  /**
     @short handle SimHit accumulation
   */
  void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);
  void accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const edm::ESHandle<HGCalGeometry> &geom, CLHEP::HepRandomEngine* hre);

  /**
     @short actions at the start/end of event
   */
  void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);

  /**
   */
  bool producesEEDigis()       { return (mySubDet_==ForwardSubdetector::HGCEE);  }
  bool producesHEfrontDigis()  { return (mySubDet_==ForwardSubdetector::HGCHEF); }
  bool producesHEbackDigis()   { return (mySubDet_==ForwardSubdetector::HGCHEB); }
  std::string digiCollection() { return digiCollection_; }

  /**
      @short actions at the start/end of run
   */
  void beginRun(const edm::EventSetup & es);
  void endRun();

private :

  //used for initialization
  bool checkValidDetIds_;

  //input/output names
  std::string hitCollection_,digiCollection_;

  //digitization type (it's up to the specializations to decide it's meaning)
  int digitizationType_;

  //handle sim hits
  int maxSimHitsAccTime_;
  double bxTime_;
  std::unique_ptr<hgc::HGCSimHitDataAccumulator> simHitAccumulator_;  
  void resetSimHitDataAccumulator();

  //digitizers
  std::unique_ptr<HGCEEDigitizer>      theHGCEEDigitizer_;
  std::unique_ptr<HGCHEbackDigitizer>  theHGCHEbackDigitizer_;
  std::unique_ptr<HGCHEfrontDigitizer> theHGCHEfrontDigitizer_;

  //subdetector id
  ForwardSubdetector mySubDet_;

  //misc switches
  bool useAllChannels_;
  uint32_t verbosity_;

  //reference speed to evaluate time of arrival at the sensititive detector, assuming the center of CMS
  float refSpeed_;

  //delay to apply after evaluating time of arrival at the sensitive detector
  float tofDelay_;
};


#endif


 
