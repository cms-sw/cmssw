#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/GFlash/interface/GFlash.h"
#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSHadronPhysicsFTFP_BERT.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4StoppingPhysics.hh"

#include "SimGeneral/GFlash/interface/GflashHistogram.h"

#include <string>

GFlash::GFlash(const edm::ParameterSet &p) : PhysicsList(p), thePar(p.getParameter<edm::ParameterSet>("GFlash")) {
  int ver = p.getUntrackedParameter<int>("Verbosity", 0);
  bool emPhys = p.getUntrackedParameter<bool>("EMPhysics", true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics", true);
  bool tracking = p.getParameter<bool>("TrackingCut");
  std::string region = p.getParameter<std::string>("Region");

  edm::LogInfo("PhysicsList") << "You are using the obsolete simulation engine: "
                              << " GFlash with Flags for EM Physics " << emPhys << ", for Hadronic Physics " << hadPhys
                              << " and tracking cut " << tracking << " with special region " << region;

  RegisterPhysics(new ParametrisedPhysics("parametrised", thePar));

  if (emPhys) {
    // EM Physics
    RegisterPhysics(new CMSEmStandardPhysics(ver, p));

    // Synchroton Radiation & GN Physics
    RegisterPhysics(new G4EmExtraPhysics(ver));
  }

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics(new G4HadronElasticPhysics(ver));

    // Hadron Physics
    RegisterPhysics(new CMSHadronPhysicsFTFP_BERT(ver));
    // Stopping Physics
    RegisterPhysics(new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics(new G4IonPhysics(ver));

    // Neutron tracking cut
    if (tracking) {
      RegisterPhysics(new G4NeutronTrackingCut(ver));
    }
  }
  // singleton histogram object
  if (thePar.getParameter<bool>("GflashHistogram")) {
    theHisto = GflashHistogram::instance();
    theHisto->setStoreFlag(true);
    theHisto->bookHistogram(thePar.getParameter<std::string>("GflashHistogramName"));
  }
}

GFlash::~GFlash() {
  if (thePar.getParameter<bool>("GflashHistogram")) {
    if (theHisto)
      delete theHisto;
  }
}

// define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"

DEFINE_PHYSICSLIST(GFlash);
