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
#include "DataFormats/HGCDigi/interface/PHGCSimAccumulator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include <vector>
#include <map>
#include <unordered_set>
#include <memory>
#include <tuple>

class PCaloHit;
class PileUpEventPrincipal;

class HGCDigitizer
{
public:
  
  HGCDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  ~HGCDigitizer() { }

  // index , det id, time
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
  template<typename GEOM>
  void accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const GEOM *geom, CLHEP::HepRandomEngine* hre);
  // for premixing
  void accumulate(const PHGCSimAccumulator& simAccumulator);

  /**
     @short actions at the start/end of event
   */
  void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine* hre);

  /**
   */
  bool producesEEDigis()       { 
    return ((mySubDet_==ForwardSubdetector::HGCEE)||(myDet_==DetId::HGCalEE)); }
  bool producesHEfrontDigis()  { 
    return ((mySubDet_==ForwardSubdetector::HGCHEF)||(myDet_==DetId::HGCalHSi)); }
  bool producesHEbackDigis()   { 
    return ((mySubDet_==ForwardSubdetector::HGCHEB)||(myDet_==DetId::HGCalHSc)); }
  std::string digiCollection() { return digiCollection_; }
  int geometryType() { return geometryType_; }

  /**
      @short actions at the start/end of run
   */
  void beginRun(const edm::EventSetup & es);
  void endRun();

private :

  uint32_t getType() const;
  bool     getWeight(std::array<float,3>& tdcForToAOnset, float& keV2fC) const;

  //input/output names
  std::string hitCollection_,digiCollection_;

  //geometry type (0 pre-TDR; 1 TDR)
  int geometryType_;

  //digitization type (it's up to the specializations to decide it's meaning)
  int digitizationType_;

  // if true, we're running mixing in premixing stage1 and have to produce the output differently
  bool premixStage1_;

  // Minimum charge threshold for premixing stage1
  double premixStage1MinCharge_;
  // Maximum charge for packing in premixing stage1
  double premixStage1MaxCharge_;

  //handle sim hits
  int maxSimHitsAccTime_;
  double bxTime_, ev_per_eh_pair_;
  std::unique_ptr<hgc::HGCSimHitDataAccumulator> simHitAccumulator_;  
  void resetSimHitDataAccumulator();

  //debug position
  void checkPosition(const HGCalDigiCollection* digis) const;

  //digitizers
  std::unique_ptr<HGCEEDigitizer>      theHGCEEDigitizer_;
  std::unique_ptr<HGCHEbackDigitizer>  theHGCHEbackDigitizer_;
  std::unique_ptr<HGCHEfrontDigitizer> theHGCHEfrontDigitizer_;
  
  //geometries
  std::unordered_set<DetId> validIds_;
  const HGCalGeometry* gHGCal_;
  const HcalGeometry* gHcal_;

  //detector and subdetector id
  DetId::Detector    myDet_;
  ForwardSubdetector mySubDet_;

  //misc switches
  uint32_t verbosity_;

  //reference speed to evaluate time of arrival at the sensititive detector, assuming the center of CMS
  float refSpeed_;

  //delay to apply after evaluating time of arrival at the sensitive detector
  float tofDelay_;

  //average occupancies
  std::array<double,3> averageOccupancies_;
  uint32_t nEvents_;

  std::vector<float> cce_;
  
  std::map< uint32_t, std::vector< std::pair<float, float> > > hitRefs_bx0;
};


#endif


 
