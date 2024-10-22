#ifndef SimG4Core_Geometry_DD4hep2DDDName_h
#define SimG4Core_Geometry_DD4hep2DDDName_h

#include <string>

namespace DD4hep2DDDName {
  std::string noNameSpace(const std::string& name);
  std::string nameMatterLV(const std::string& name, bool dd4hep);
  std::string nameSolid(const std::string& name, bool dd4hep);
  std::string namePV(const std::string& name, bool dd4hep);
};  // namespace DD4hep2DDDName

#endif
