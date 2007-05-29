#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "SimG4Core/Physics/interface/DDG4ProductionCuts.h"

#ifndef G4V7
#include "G4LossTableManager.hh"
#endif

PhysicsList::PhysicsList(G4LogicalVolumeToDDLogicalPartMap & map,
			 const edm::ParameterSet & p) 
  : G4VModularPhysicsList(), map_(map), m_pPhysics(p),  prodCuts(0)
{
  //if (m_pPhysics.getParameter<bool>("CutsPerRegion")) 
  // prodCuts = new DDG4ProductionCuts();	
}
 
PhysicsList::~PhysicsList() 
{
#ifndef G4V7
  if (m_pPhysics.getUntrackedParameter<int>("Verbosity",0) > 1)
    std::cout << " G4BremsstrahlungThreshold was " 
	      << G4LossTableManager::Instance()->BremsstrahlungTh()/GeV 
	      << " GeV " << std::endl;
#endif
  if (prodCuts!=0) delete prodCuts;
}

void PhysicsList::SetCuts() 
{ 

  SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*cm);
  SetCutsWithDefault();

#ifndef G4V7
  G4LossTableManager::Instance()->SetBremsstrahlungTh
    (m_pPhysics.getParameter<double>("G4BremsstrahlungThreshold")*GeV);
#endif
    
  int v =  m_pPhysics.getUntrackedParameter<int>("Verbosity",0);
  if ( m_pPhysics.getParameter<bool>("CutsPerRegion") )
    {
      DDG4ProductionCuts prodCuts(map_);
      prodCuts.SetVerbosity(v);
      prodCuts.update();
    }

  if ( v > 1) {
#ifndef G4V7
    G4LossTableManager::Instance()->SetVerbose(v-1);
#endif
    G4VUserPhysicsList::DumpCutValuesTable();
  }

  return ;

}

