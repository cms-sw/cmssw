#include "SimG4Core/GFlash/interface/GFlash.h"
#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"
#include "SimG4Core/GFlash/interface/HadronPhysicsQGSP_WP.h"
#include "SimG4Core/GFlash/interface/HadronPhysicsQGSP_BERT_WP.h"
#include "SimG4Core/GFlash/interface/HadronPhysicsQGSPCMS_FTFP_BERT_WP.h"
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
	       sim::FieldBuilder *fieldBuilder_, 
	       const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p), 
					      thePar(p.getParameter<edm::ParameterSet>("GFlash")) {

  G4DataQuestionaire it(photon);

  //std::string hadronPhysics = thePar.getParameter<std::string>("GflashHadronPhysics");

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  std::string region = p.getParameter<std::string>("Region");
  bool gem  = thePar.getParameter<bool>("GflashEMShowerModel");
  bool ghad = thePar.getParameter<bool>("GflashHadronShowerModel");

  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << " + CMS GFLASH with Flags for EM Physics "
                              << gem << ", for Hadronic Physics "
			      << ghad 
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
    /*
    if(hadronPhysics=="QGSP_FTFP_BERT") {
      RegisterPhysics( new HadronPhysicsQGSPCMS_FTFP_BERT_WP("hadron",quasiElastic)); 
    }
    else if(hadronPhysics=="QGSP_BERT") {
      RegisterPhysics( new HadronPhysicsQGSP_BERT_WP("hadron",quasiElastic));
    }
    else if (hadronPhysics=="QGSP") {
      RegisterPhysics( new HadronPhysicsQGSP_WP("hadron",quasiElastic));
    }
    else {
      edm::LogInfo("PhysicsList") << hadronPhysics << " is not available for GflashHadronPhysics!"
				  << "... Using QGSP_FTFP_BERT\n";
      RegisterPhysics( new HadronPhysicsQGSPCMS_FTFP_BERT_WP("hadron",quasiElastic));
    }
    // Stopping Physics
    RegisterPhysics( new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));

    // Neutron tracking cut
    if (tracking) 
      RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));
    */
  }

  // Monopoles
  RegisterPhysics( new CMSMonopolePhysics(table_,fieldBuilder_,p));

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

