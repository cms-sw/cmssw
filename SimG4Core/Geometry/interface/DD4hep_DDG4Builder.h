#ifndef SimG4Core_DD4hep_DDG4Builder_h
#define SimG4Core_DD4hep_DDG4Builder_h

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DDG4/Geant4Converter.h"
#include "DDG4/Geant4GeometryInfo.h"
#include "DDG4/Geant4Mapping.h"
#include "DD4hep/Detector.h"
#include "DD4hep/Printout.h"

#include <map>
#include <string>
#include <vector>

namespace cms {
  class DDCompactView;
}

class SensitiveDetectorCatalog;

namespace cms {
  class DDG4Builder {
  public:
    DDG4Builder(const cms::DDCompactView *, dd4hep::sim::Geant4GeometryMaps::VolumeMap &, bool check);
    G4VPhysicalVolume *BuildGeometry(SensitiveDetectorCatalog &);

  private:
    const cms::DDCompactView *compactView_;
    dd4hep::sim::Geant4GeometryMaps::VolumeMap &map_;
    bool check_;
  };
}  // namespace cms

#endif
