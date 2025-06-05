///////////////////////////////////////////////////////////////////////////////
// File: ZdcTestTreeAnalysis.cc
// Date: 04.25 Lev Kheyn
// Description: simulation analysis code to make a tree needed for making
//              shower library for ZDC
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <CLHEP/Random/Randomize.h>

#include "TFile.h"
#include "TTree.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

class ZdcTestTreeAnalysis : public SimWatcher,
                            public Observer<const BeginOfJob*>,
                            public Observer<const BeginOfRun*>,
                            public Observer<const EndOfRun*>,
                            public Observer<const BeginOfEvent*>,
                            public Observer<const EndOfEvent*>,
                            public Observer<const G4Step*> {
public:
  ZdcTestTreeAnalysis(const edm::ParameterSet& p);
  ~ZdcTestTreeAnalysis() override;

private:
  // observer classes
  void update(const BeginOfJob* run) override;
  void update(const BeginOfRun* run) override;
  void update(const EndOfRun* run) override;
  void update(const BeginOfEvent* evt) override;
  void update(const EndOfEvent* evt) override;
  void update(const G4Step* step) override;

  int verbosity_;
  TTree* theTree;
  int eventIndex, nhits;
  int fiberID[2000], npeem[2000], npehad[2000], time[2000];
};

ZdcTestTreeAnalysis::ZdcTestTreeAnalysis(const edm::ParameterSet& p) {
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("ZdcTestTreeAnalysis");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");
}

ZdcTestTreeAnalysis::~ZdcTestTreeAnalysis() {}

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
void ZdcTestTreeAnalysis::update(const BeginOfJob* job) {
  //job
  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << "ZdcTestTreeAnalysis::Beggining of job";
  edm::Service<TFileService> theFile;
  theTree = theFile->make<TTree>("CherenkovPhotons", "Cherenkov Photons");
  theTree->Branch("nhits", &nhits, "nhits/I");
  theTree->Branch("fiberID", fiberID, "fiberID/I");
  theTree->Branch("npeem", npeem, "npeem/I");
  theTree->Branch("npehad", npehad, "npehad/I");
  theTree->Branch("time", time, "time/I");
};

//==================================================================== per RUN
void ZdcTestTreeAnalysis::update(const BeginOfRun* run) {
  //run

  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << "\nZdcTestTreeAnalysis: Begining of Run";

  eventIndex = 0;
}

void ZdcTestTreeAnalysis::update(const BeginOfEvent* evt) {
  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << "ZdcTest: Processing Event Number: " << eventIndex;
  eventIndex++;
}

//================================================================================================
void ZdcTestTreeAnalysis::update(const G4Step* aStep) {}

//================================================================================================
void ZdcTestTreeAnalysis::update(const EndOfEvent* evt) {
  // access to the G4 hit collections
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << "  accessed all HC";

  int theZDCHCid = G4SDManager::GetSDMpointer()->GetCollectionID("ZDCHITS");
  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << " - theZDCHCid = " << theZDCHCid;

  CaloG4HitCollection* theZDCHC = (CaloG4HitCollection*)allHC->GetHC(theZDCHCid);
  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << " - theZDCHC = " << theZDCHC;

  int nentries = theZDCHC->entries();
  if (verbosity_ > 0)
    edm::LogVerbatim("ZdcTestTreeAnalysis") << "  theZDCHC has " << nentries << " entries";

  if (nentries > 0) {
    for (int ihit = 0; ihit < nentries; ihit++) {
      CaloG4Hit* aHit = (*theZDCHC)[ihit];
      fiberID[ihit] = aHit->getUnitID();
      npeem[ihit] = aHit->getEM();
      npehad[ihit] = aHit->getHadr();
      time[ihit] = aHit->getTimeSliceID();

      if (verbosity_ > 1)
        edm::LogVerbatim("ZdcTestTreeAnalysis")
            << " entry #" << ihit << ": fiaberID=0x" << std::hex << fiberID[ihit] << std::dec
            << "; npeem=" << npeem[ihit] << "; npehad[ihit]=" << npehad << " time=" << time[ihit];
    }
  }
  nhits = nentries;
  theTree->Fill();
}

void ZdcTestTreeAnalysis::update(const EndOfRun* run) {}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(ZdcTestTreeAnalysis);
