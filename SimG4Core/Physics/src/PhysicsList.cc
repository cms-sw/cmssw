#include "SimG4Core/Physics/interface/PhysicsList.h"
#include "SimG4Core/Physics/interface/DDG4ProductionCuts.h"

PhysicsList::PhysicsList(const edm::ParameterSet & p) 
  : G4VModularPhysicsList(), m_pPhysics(p),  prodCuts(0)
{
    //if (m_pPhysics.getParameter<bool>("CutsPerRegion")) 
      // prodCuts = new DDG4ProductionCuts();	
}
 
PhysicsList::~PhysicsList() 
{
    if (prodCuts!=0) delete prodCuts;
}

void PhysicsList::SetCuts() 
{ 

    SetDefaultCutValue(m_pPhysics.getParameter<double>("DefaultCutValue")*cm);
    SetCutsWithDefault();
    
    if ( m_pPhysics.getParameter<bool>("CutsPerRegion") )
    {
       DDG4ProductionCuts prodCuts;
       prodCuts.update();
    }

    if (m_pPhysics.getParameter<int>("Verbosity") > 1) 
	G4VUserPhysicsList::DumpCutValuesTable();

    return ;

}

