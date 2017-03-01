#include "SimG4Core/CustomPhysics/interface/CustomPhysics.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsList.h"
#include "SimG4Core/CustomPhysics/interface/CustomPhysicsListSS.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics95msc93.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
#include "G4HadronPhysicsQGSP_FTFP_BERT.hh"
#include "G4SystemOfUnits.hh"
 
CustomPhysics::CustomPhysics(G4LogicalVolumeToDDLogicalPartMap& map, 
			     const HepPDT::ParticleDataTable * table_,
			     sim::ChordFinderSetter *chordFinderSetter_, 
			     const edm::ParameterSet & p) : PhysicsList(map, table_, chordFinderSetter_, p) {

  G4DataQuestionaire it(photon);

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool tracking= p.getParameter<bool>("TrackingCut");
  bool ssPhys  = p.getUntrackedParameter<bool>("ExoticaPhysicsSS",false);
  double timeLimit = p.getParameter<double>("MaxTrackTime")*ns;
  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << "QGSP_FTFP_BERT_EML for regular particles \n"
			      << "CustomPhysicsList " << ssPhys << " for exotics; "
                              << " tracking cut " << tracking << "  t(ns)= " << timeLimit/ns;
  // EM Physics
  RegisterPhysics(new CMSEmStandardPhysics(ver));
  //RegisterPhysics(new CMSEmStandardPhysics95msc93("EM standard msc93",ver,""));

  // Synchroton Radiation & GN Physics
  RegisterPhysics(new G4EmExtraPhysics(ver));

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics(ver)); 

  // Hadron Physics
  RegisterPhysics(new G4HadronPhysicsQGSP_FTFP_BERT(ver));

  // Stopping Physics
  RegisterPhysics(new G4StoppingPhysics(ver));

  // Ion Physics
  RegisterPhysics(new G4IonPhysics(ver));

  // Neutron tracking cut
  if (tracking) {
    G4NeutronTrackingCut* ncut= new G4NeutronTrackingCut(ver);
    ncut->SetTimeLimit(timeLimit);
    RegisterPhysics(ncut);
  }

  // Custom Physics
  if(ssPhys) {
    RegisterPhysics(new CustomPhysicsListSS("custom",p));
  } else {
    RegisterPhysics(new CustomPhysicsList("custom",p));    
  }
}
