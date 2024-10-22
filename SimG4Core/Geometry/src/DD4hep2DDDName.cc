#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"

std::string DD4hep2DDDName::noNameSpace(const std::string& name) {
  std::size_t found = name.find(':');
  std::string nam = (found == std::string::npos) ? name : name.substr(found + 1, (name.size() - found));
  return nam;
}

std::string DD4hep2DDDName::nameMatterLV(const std::string& name, bool dd4hep) {
  return (dd4hep ? (DD4hep2DDDName::noNameSpace(name)) : name);
}

std::string DD4hep2DDDName::nameSolid(const std::string& name, bool dd4hep) {
  if (!dd4hep)
    return name;
  std::string nam = DD4hep2DDDName::noNameSpace(name);
  auto n = nam.find("_shape");
  if (n != std::string::npos)
    nam = nam.substr(0, n);
  if (name.find("_refl") != std::string::npos)
    nam += "_refl";
  return nam;
}

std::string DD4hep2DDDName::namePV(const std::string& name, bool dd4hep) {
  if (!dd4hep)
    return name;
  std::string nam = DD4hep2DDDName::noNameSpace(name);
  auto n = nam.rfind('_');
  return ((n != std::string::npos) ? nam.substr(0, n) : nam);
}
