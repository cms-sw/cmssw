#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"
#include <DD4hep/Filter.h>

std::string_view DD4hep2DDDName::nameMatterLV(const std::string& name, bool dd4hep) {
  return (dd4hep ? (dd4hep::dd::noNamespace(name)) : name);
}

std::string DD4hep2DDDName::nameSolid(const std::string& name, bool dd4hep) {
  if (!dd4hep)
    return name;
  std::string nam = static_cast<std::string>(dd4hep::dd::noNamespace(name));
  auto n = nam.find("_shape");
  if (n != std::string::npos)
    nam = nam.substr(0, n);
  if (name.find("_refl") != std::string::npos)
    nam += "_refl";
  return nam;
}

std::string_view DD4hep2DDDName::namePV(const std::string& name, bool dd4hep) {
  if (!dd4hep)
    return name;
  std::string_view nam = (dd4hep::dd::noNamespace(name));
  auto n = nam.rfind('_');
  return ((n != std::string::npos) ? nam.substr(0, n) : nam);
}
