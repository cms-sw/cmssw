#include "SimG4Core/GFlash/interface/GFlash.h"
#include "SimG4Core/GFlash/interface/ParametrisedPhysics.h"
#include "SimG4Core/GFlash/interface/HadronPhysicsQGSP_WP.h"
#include "SimG4Core/GFlash/interface/HadronPhysicsQGSP_BERT_WP.h"
#include "SimG4Core/GFlash/interface/HadronPhysicsQGSPCMS_FTFP_BERT_WP.h"
#include "SimG4Core/PhysicsLists/interface/HadronPhysicsQGSPCMS_FTFP_BERT.h"
#include "SimG4Core/PhysicsLists/interface/CMSEmStandardPhysics92.h"
#include "SimG4Core/PhysicsLists/interface/CMSMonopolePhysics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4QStoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh" 
#include "G4NeutronTrackingCut.hh"

#include "G4DataQuestionaire.hh"
#include "SimGeneral/GFlash/interface/GflashHistogram.h"

#include <string>

GFlash::GFlash(G4LogicalVolumeToDDLogicalPartMap& map, 
	       const HepPDT::ParticleDataTable * table_, 
	       sim::FieldBuilder *fieldBuilder_, 
	       const edm::ParameterSet & p) : PhysicsList(map, table_, fieldBuilder_, p), 
					      thePar(p.getParameter<edm::ParameterSet>("GFlash")) {

  G4DataQuestionaire it(photon);

  std::string hadronPhysics = thePar.getParameter<std::string>("GflashHadronPhysics");

  int  ver     = p.getUntrackedParameter<int>("Verbosity",0);
  bool emPhys  = p.getUntrackedParameter<bool>("EMPhysics",true);
  bool hadPhys = p.getUntrackedParameter<bool>("HadPhysics",true);
  bool tracking= p.getParameter<bool>("TrackingCut");
  std::string region = p.getParameter<std::string>("Region");

  edm::LogInfo("PhysicsList") << "You are using the simulation engine: "
			      << hadronPhysics << " + CMS GFLASH with Flags for EM Physics "
                              << emPhys << ", for Hadronic Physics "
                              << hadPhys << " and tracking cut " << tracking
                              << " with special region " << region;

  RegisterPhysics(new ParametrisedPhysics("parametrised",thePar)); 

  if (emPhys) {
    // EM Physics
    RegisterPhysics( new CMSEmStandardPhysics92("standard EM EML",ver,region));

    // Synchroton Radiation & GN Physics
    RegisterPhysics( new G4EmExtraPhysics("extra EM"));
  }

  // Decays
  RegisterPhysics( new G4DecayPhysics("decay",ver) );

  if (hadPhys) {
    // Hadron Elastic scattering
    RegisterPhysics( new G4HadronElasticPhysics("elastic",ver,false));

    // Hadron Physics
    G4bool quasiElastic=true;
    RegisterPhysics( new HadronPhysicsQGSPCMS_FTFP_BERT("hadron",quasiElastic));   
    // Stopping Physics
    RegisterPhysics( new G4QStoppingPhysics("stopping"));

    // Ion Physics
    RegisterPhysics( new G4IonPhysics("ion"));

    // Neutron tracking cut
    if (tracking) 
      RegisterPhysics( new G4NeutronTrackingCut("Neutron tracking cut", ver));

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

