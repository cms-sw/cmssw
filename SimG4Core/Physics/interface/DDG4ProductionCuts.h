#ifndef SimG4Core_DDG4ProductionCuts_H
#define SimG4Core_DDG4ProductionCuts_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"

#include <string>
#include <vector>

class DDLogicalPart;
class G4Region;
class G4LogicalVolume;
class G4ProductionCuts;

class DDG4ProductionCuts  {

public:
  DDG4ProductionCuts(const G4LogicalVolumeToDDLogicalPartMap&, int,
		     const edm::ParameterSet & p);
  ~DDG4ProductionCuts();
  void update();
  void SetVerbosity( int verb ) { m_Verbosity = verb; return ; }

private:

  void initialize();
  void setProdCuts(const DDLogicalPart lpart, G4LogicalVolume * lvolume);
  G4Region * getRegion(const std::string & region);
  G4ProductionCuts * getProductionCuts(G4Region * region);

  G4LogicalVolumeToDDLogicalPartMap map_;
  std::string                       m_KeywordRegion;    
  int                               m_Verbosity;
  bool                              m_protonCut;
  G4LogicalVolumeToDDLogicalPartMap::Vector vec_;    
};

#endif
