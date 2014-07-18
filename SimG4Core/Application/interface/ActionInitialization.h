#ifndef SimG4Core_ActionInitialization_h
#define SimG4Core_ActionInitialization_h 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Generators/interface/Generator.h"

#include "G4VUserActionInitialization.hh"

// Action initialization class.
//

class RunManager;
class CMSRunInterface;

class ActionInitialization : public G4VUserActionInitialization
{
public:

  ActionInitialization(const edm::ParameterSet & ps, 
		       RunManager* runm);

  virtual ~ActionInitialization();

  virtual void BuildForMaster() const;
  virtual void Build() const;

private:

  RunManager*       m_runManager;

  edm::ParameterSet m_pGenerator;   
  edm::ParameterSet m_pVertexGenerator;
  edm::ParameterSet m_pPhysics; 
  edm::ParameterSet m_pRunAction;      
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;

};

#endif

    
