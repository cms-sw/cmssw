#ifndef SimG4CMS_CaloSteppingAction_H
#define SimG4CMS_CaloSteppingAction_H
//#define HcalNumberingTest

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#ifndef useGeantV
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#endif

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"

#include "SimG4CMS/Calo/interface/CaloGVHit.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalNumberingFromPS.h"

#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#ifdef HcalNumberingTest
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#endif

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef useGeantV
// GeantV headers
#include <Geant/Track.h>
#include <Geant/Typedefs.h>
#include <Geant/SystemOfUnits.h>
#include <Geant/MaterialCuts.h>
#include <Geant/Region.h>

// VecGeom headers
#include <management/GeoManager.h>
#include <volumes/PlacedVolume.h>
#include <volumes/LogicalVolume.h>
#else
// Geant4 headers
#include "G4LogicalVolume.hh"
#include "G4Region.hh"
#include "G4Step.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Track.hh"
#endif

#ifdef useGeantV
class CaloSteppingAction {
#else
class CaloSteppingAction : public SimProducer,
                           public Observer<const BeginOfJob *>, 
                           public Observer<const BeginOfRun *>, 
                           public Observer<const BeginOfEvent *>, 
                           public Observer<const EndOfEvent *>, 
                           public Observer<const G4Step *> {
#endif
public:
  CaloSteppingAction(const edm::ParameterSet &p);
#ifdef useGeantV
  ~CaloSteppingAction();

  void beginRun();
  void beginEvent(int evt);
  void steppinAction(Track& );
  void endEvent(Event* );

private:  
#else
  ~CaloSteppingAction() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // observer classes
  void update(const BeginOfJob * job)   override;
  void update(const BeginOfRun * run)   override;
  void update(const BeginOfEvent * evt) override;
  void update(const G4Step * step)      override;
  void update(const EndOfEvent * evt)   override;

  void NaNTrap(const G4Step*) const;
#endif
  void fillHits(edm::PCaloHitContainer& cc, int type);
  uint32_t getDetIDHC(int det, int lay, int depth,
		      const math::XYZVectorD& pos) const;
  void fillHit(uint32_t id, double dE, double time, int primID, 
	       uint16_t depth, double em, int flag);
  uint16_t getDepth(bool flag, double crystalDepth, double radl) const;
  double   curve_LY(double crystalLength, double crystalDepth) const;
  double   getBirkL3(double dE, double step, double chg, double dens) const;
  double   getBirkHC(double dE, double step, double chg, double dens) const;
  void     saveHits(int flag);

  static const int                      nSD_= 3;
  std::unique_ptr<EcalBarrelNumberingScheme> ebNumberingScheme_;
  std::unique_ptr<EcalEndcapNumberingScheme> eeNumberingScheme_;
  std::unique_ptr<HcalNumberingFromPS>       hcNumberingPS_;
#ifdef HcalNumberingTest
  std::unique_ptr<HcalNumberingFromDDD>      hcNumbering_;
#endif
  std::unique_ptr<HcalNumberingScheme>       hcNumberingScheme_;
  std::unique_ptr<CaloSlaveSD>               slave_[nSD_];

  std::vector<std::string>              nameEBSD_, nameEESD_, nameHCSD_;
  std::vector<std::string>              nameHitC_;
#ifdef useGeantV
  std::vector<const Volume_t>           volEBSD_, volEESD_, volHCSD_;
  std::map<const Volume_t,double>       xtalMap_;
#else
  std::vector<const G4LogicalVolume*>   volEBSD_, volEESD_, volHCSD_;
  std::map<const G4LogicalVolume*,double> xtalMap_;
#endif
  int                                   count_, eventID_;
  double                                slopeLY_, birkC1EC_, birkSlopeEC_;
  double                                birkCutEC_, birkC1HC_, birkC2HC_;
  double                                birkC3HC_;
  std::map<std::pair<int,CaloHitID>,CaloGVHit> hitMap_[nSD_];
};

#endif
