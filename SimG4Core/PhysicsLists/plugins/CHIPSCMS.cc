#include "CHIPSCMS.hh"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4QPhotoNuclearPhysics.hh"
#include "G4QNeutrinoPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QCaptureAtRestPhysics.hh"
#include "G4HadronQElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
#include "HadronPhysicsCHIPS.hh"

CHIPSCMS::CHIPSCMS(G4LogicalVolumeToDDLogicalPartMap& map,
		   const HepPDT::ParticleDataTable * table_,
		   sim::FieldBuilder *fieldBuilder_,
		   const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p) {

  G4DataQuestionaire it(photon);
  
  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "CHIPS 1.0 with Flags for EM Physics "
                              << emPhys << ", for Hadronic Physics "
                              << hadPhys << " and tracking cut " << tracking;

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics("standard EM",ver));

    // Synchroton Radiation & Photo-Nuclear Physics
    RegisterPhysics( new G4QPhotoNuclearPhysics("photo-nuclear"));
  }

  // Neutrino-Nuclear Physics
  RegisterPhysics( new G4QNeutrinoPhysics("weak"));

  // Decays
  RegisterPhysics(new G4DecayPhysics("decay",ver));

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronQElasticPhysics("elastic",ver));

    // Hadron Physics (to be replaced by G4QInelasticPhysics)
    RegisterPhysics(  new HadronPhysicsCHIPS("inelastic"));

    // Stopping Physics
    RegisterPhysics( new G4QCaptureAtRestPhysics("nuclear_capture",ver));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));

    // Neutron tracking cut
    if (tracking)
      RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));
  }
}

