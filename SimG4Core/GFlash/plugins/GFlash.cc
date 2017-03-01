#include "SimG4Core/GFlash/interface/GFlash.h"
#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics95msc93.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4NeutronTrackingCut.hh"
#include "G4HadronPhysicsQGSP_FTFP_BERT.hh"

#include "G4DataQuestionaire.hh"
#include "SimGeneral/GFlash/interface/GflashHistogram.h"

#include <string>

GFlash::GFlash(G4LogicalVolumeToDDLogicalPartMap& map, 
	       const HepPDT::ParticleDataTable * table_, 
	       sim::ChordFinderSetter *chordFinderSetter_, 
	       const edm::ParameterSet & p) : PhysicsList(map, table_, chordFinderSetter_, p), 
					      thePar(p.getParameter<edm::ParameterSet>("GFlash")) {

  G4DataQuestionaire it(photon);

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  std::string region = p.getParameter<std::string>("Region");

  edm::LogInfo("PhysicsList") << "You are using the obsolete simulation engine: "
			      << " GFlash with Flags for EM Physics "
                              << emPhys << ", for Hadronic Physics "
			      << hadPhys
                              << " and tracking cut " << tracking
                              << " with special region " << region;

  RegisterPhysics(new ParametrisedPhysics("parametrised",thePar)); 

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics95msc93("EM standard msc93",ver,region));

    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics(ver));
  }

  // Decays
  RegisterPhysics( new G4DecayPhysics(ver) );

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics(ver));

    // Hadron Physics
    RegisterPhysics( new G4HadronPhysicsQGSP_FTFP_BERT(ver));   
    // Stopping Physics
    RegisterPhysics( new G4StoppingPhysics(ver));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics(ver));

    // Neutron tracking cut
    if (tracking) {
      RegisterPhysics( new G4NeutronTrackingCut(ver));
    }
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,chordFinderSetter_,p));

  // singleton histogram object
  if(thePar.getParameter<bool>("GflashHistogram")) {
    theHisto = GflashHistogram::instance();
    theHisto->setStoreFlag(true);
    theHisto->bookHistogram(thePar.getParameter<std::string>("GflashHistogramName"));
  }
}

GFlash::~GFlash() {

  if(thePar.getParameter<bool>("GflashHistogram")) {
    if(theHisto) delete theHisto;
  }

}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimG4Core/Physics/interface/PhysicsListFactory.h"


DEFINE_PHYSICSLIST(GFlash);

