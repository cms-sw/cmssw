///////////////////////////////////////////////////////////////////////////////
// File: HGCPassive.cc
//copied from SimG4HGCalValidation
// Description: Main analysis class for HGCal Validation of G4 Hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

// to retreive hits
#include "SimDataFormats/CaloHit/interface/PassiveHit.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/CaloHit/interface/PassiveHit.h"

#include "G4Step.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"

#include <vector>
#include <string>
#include <map>

class HGCPassive : public SimProducer,
		   public Observer<const BeginOfRun *>, 
		   public Observer<const BeginOfEvent *>, 
		   public Observer<const G4Step *> {

  
public:
  HGCPassive(const edm::ParameterSet &p);
  virtual ~HGCPassive();

  void produce(edm::Event&, const edm::EventSetup&);

private:
  HGCPassive(const HGCPassive&); // stop default
  const HGCPassive& operator=(const HGCPassive&);

  // observer classes
  void update(const BeginOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const G4Step * step);
  
  //void endOfEvent(edm::PassiveHitContainer &HGCEEAbsE);
  void endOfEvent(edm::PassiveHitContainer& hgcPH, unsigned int k);

  typedef std::map<G4LogicalVolume*,std::pair<unsigned int,std::string>>::iterator volumeIterator;
  G4VPhysicalVolume * getTopPV();
  volumeIterator      findLV(G4LogicalVolume * plv);
  void storeInfo(const volumeIterator itr, G4LogicalVolume* plv, 
		 unsigned int copy, double time, double energy);

private:

  std::vector<std::string> LVNames_;
  G4VPhysicalVolume       *topPV_; 
  G4LogicalVolume         *topLV_;
  std::map<G4LogicalVolume*,std::pair<unsigned int,std::string>> mapLV_;
  std::string              motherName_;
  
  // some private members for ananlysis 
  unsigned int              count_;                  
  bool                      init_;
  std::map<std::pair<G4LogicalVolume*,unsigned int>,std::pair<double,double>> store_;
};


