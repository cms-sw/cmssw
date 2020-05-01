#ifndef SimG4Core_MagneticField_FieldBuilder_H
#define SimG4Core_MagneticField_FieldBuilder_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class CMSFieldManager;
class G4Mag_UsualEqRhs;
class G4PropagatorInField;
class G4LogicalVolume;

namespace local {
  class MagneticField;
}

namespace sim {
  class Field;
  class FieldBuilder {
  public:
    FieldBuilder(local::MagneticField, const edm::ParameterSet &);

    ~FieldBuilder();

    void build(CMSFieldManager *fM, G4PropagatorInField *fP);

    void configureForVolume(const std::string &volName,
                            edm::ParameterSet &volPSet,
                            CMSFieldManager *fM,
                            G4PropagatorInField *fP);

  private:
    Field *theField;
    G4Mag_UsualEqRhs *theFieldEquation;
    G4LogicalVolume *theTopVolume;
    edm::ParameterSet thePSet;
    double theDelta;
  };
};  // namespace sim

#endif
