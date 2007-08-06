#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "SimG4Core/Physics/interface/DDG4ProductionCuts.h"

#include "G4LossTableManager.hh"

PhysicsList::PhysicsList(G4LogicalVolumeToDDLogicalPartMap & map,
			 const edm::ParameterSet & p) 
  : G4VModularPhysicsList(), map_(map), m_pPhysics(p),  prodCuts(0)
{
  //if (m_pPhysics.getParameter<bool>("CutsPerRegion")) 
  // prodCuts = new DDG4ProductionCuts();	
}
 
PhysicsList::~PhysicsList() 
{
  if (m_pPhysics.getUntrackedParameter<int>("Verbosity",0) > 1)
    std::cout << " G4BremsstrahlungThreshold was " 
	      << G4LossTableManager::Instance()->BremsstrahlungTh()/GeV 
	      << " GeV " << std::endl;
  if (prodCuts!=0) delete prodCuts;
}

void PhysicsList::SetCuts() 
{ 

  SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*cm);
  SetCutsWithDefault();

  G4LossTableManager::Instance()->SetBremsstrahlungTh
    (m_pPhysics.getParameter<double>("G4BremsstrahlungThreshold")*GeV);

  int v =  m_pPhysics.getUntrackedParameter<int>("Verbosity",0);
  if ( m_pPhysics.getParameter<bool>("CutsPerRegion") )
    {
      DDG4ProductionCuts prodCuts(map_);
      prodCuts.SetVerbosity(v);
      prodCuts.update();
    }

  if ( v > 1) {
    G4LossTableManager::Instance()->SetVerbose(v-1);
    G4VUserPhysicsList::DumpCutValuesTable();
  }

  return ;

}

