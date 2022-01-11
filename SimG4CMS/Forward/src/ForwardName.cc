#include "SimG4CMS/Forward/interface/ForwardName.h"

std::string ForwardName::getName(const G4String& namx) {
  std::string name = static_cast<std::string>(namx);
  if (name.find(':') == std::string::npos) {
    return name;
  } else {
    std::size_t first = name.find(':') + 1;
    std::size_t last = name.rfind('_');
    std::size_t length = (last != std::string::npos) ? (last - first) : (name.size() - first);
    return name.substr(first, length);
  }
}
