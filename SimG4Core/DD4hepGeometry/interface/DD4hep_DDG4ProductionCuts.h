#ifndef SimG4Core_DD4hep_DDG4ProductionCuts_H
#define SimG4Core_DD4hep_DDG4ProductionCuts_H

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DDG4/Geant4GeometryInfo.h"

#include <string>
#include <vector>

class G4Region;
class G4LogicalVolume;
class G4ProductionCuts;

class DD4hep_DDG4ProductionCuts {

 public:
  DD4hep_DDG4ProductionCuts(const cms::DDSpecParRegistry*,
			    const dd4hep::sim::Geant4GeometryMaps::VolumeMap&, int,
			    const edm::ParameterSet&);
  ~DD4hep_DDG4ProductionCuts();
  void update();
  void setVerbosity( int verb ) { m_verbosity = verb; }
  
 private:
  
  void initialize();
  void setProdCuts(G4LogicalVolume* lvolume);
  G4Region * getRegion(const std::string & region);
  G4ProductionCuts * getProductionCuts(G4Region * region);

  const cms::DDSpecParRegistry*     m_specPars;
  const dd4hep::sim::Geant4GeometryMaps::VolumeMap& m_map;
  cms::DDSpecParRefs                m_specs;
  std::vector<std::pair<G4LogicalVolume*, const cms::DDSpecPar*>> m_vec;
  std::string_view                  m_keywordRegion;    
  int                               m_verbosity;
  bool                              m_protonCut;
};

#endif
