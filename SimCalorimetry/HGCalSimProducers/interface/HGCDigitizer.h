#ifndef HGCalSimProducers_HGCDigitizer_h
#define HGcalSimProducers_HGCDigitizer_h

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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include <vector>
#include <map>
#include <memory>

class PCaloHit;
class PileUpEventPrincipal;

namespace edm {
  class ConsumesCollector;
}

namespace CLHEP {
  class HepRandomEngine;
}

class HGCDigitizer {

public:
  
  explicit HGCDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);
  ~HGCDigitizer() { }

  /**
     @short handle SimHit accumulation
   */
  void accumulate(edm::Event const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine*);
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, CLHEP::HepRandomEngine*);
  void accumulate(edm::Handle<edm::PCaloHitContainer> const &hits, int bxCrossing,const edm::ESHandle<HGCalGeometry> &geom, CLHEP::HepRandomEngine*);

  /**
     @short actions at the start/end of event
   */
  void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c, CLHEP::HepRandomEngine*);

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
  int bxTime_;
  std::unique_ptr<HGCSimHitDataAccumulator> simHitAccumulator_;  
  void resetSimHitDataAccumulator();

  //digitizers
  std::unique_ptr<HGCEEDigitizer>      theHGCEEDigitizer_;
  std::unique_ptr<HGCHEbackDigitizer>  theHGCHEbackDigitizer_;
  std::unique_ptr<HGCHEfrontDigitizer> theHGCHEfrontDigitizer_;

  //subdetector id
  ForwardSubdetector mySubDet_;

  //misc switches
  bool useAllChannels_;
  int  verbosity_;

  //reference speed to evaluate time of arrival at the sensititive detector, assuming the center of CMS
  float refSpeed_;

  //delay to apply after evaluating time of arrival at the sensitive detector
  float tofDelay_;
};

#endif
