#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "SimG4Core/Physics/interface/DDG4ProductionCuts.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4LossTableManager.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList(G4LogicalVolumeToDDLogicalPartMap & map,
			 const HepPDT::ParticleDataTable * table_,
			 sim::ChordFinderSetter *chordFinderSetter_,
			 const edm::ParameterSet & p) 
  : G4VModularPhysicsList(), m_pPhysics(p),  prodCuts(0) {
  m_Verbosity = m_pPhysics.getUntrackedParameter<int>("Verbosity",0);
  prodCuts = new DDG4ProductionCuts(map, m_Verbosity, m_pPhysics);	
}
 
PhysicsList::~PhysicsList() {
  delete prodCuts;
}

void PhysicsList::SetCuts() { 

  SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*cm);
  SetCutsWithDefault();

  if ( m_pPhysics.getParameter<bool>("CutsPerRegion") ) {
    prodCuts->update();
  }

  if ( m_Verbosity > 1) {
    G4VUserPhysicsList::DumpCutValuesTable();
  }

  return ;

}

