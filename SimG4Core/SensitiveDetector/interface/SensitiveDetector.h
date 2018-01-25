#ifndef SimG4Core_SensitiveDetector_H
#define SimG4Core_SensitiveDetector_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
//#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "G4VSensitiveDetector.hh"

#include <string>

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;
class G4VPhysicalVolume;
class DDCompactView;    

class SensitiveDetector : public G4VSensitiveDetector
{
public:
  explicit SensitiveDetector(const std::string & iname, 
                             const DDCompactView & cpv,
			     const SensitiveDetectorCatalog & ,
			     edm::ParameterSet const & p);
  ~SensitiveDetector() override;

  void Initialize(G4HCofThisEvent * eventHC) override;
  G4bool ProcessHits(G4Step * step ,G4TouchableHistory * tHistory) override = 0;
  void EndOfEvent(G4HCofThisEvent * eventHC) override;

  virtual uint32_t setDetUnitId(const G4Step * step) = 0;
  virtual void clearHits() = 0;

  inline const std::vector<std::string>& getNames() const { return namesOfSD; }
 
protected:

  enum coordinates {WorldCoordinates, LocalCoordinates};
  Local3DPoint InitialStepPosition(const G4Step * step, coordinates) const;
  Local3DPoint FinalStepPosition(const G4Step * step, coordinates) const;

  Local3DPoint LocalPreStepPosition(const G4Step * step) const;
  Local3DPoint LocalPostStepPosition(const G4Step * step) const;

  inline Local3DPoint ConvertToLocal3DPoint(const G4ThreeVector& point) const
  { return Local3DPoint(point.x(),point.y(),point.z()); }
  
  void setNames(const std::vector<std::string>&);
  void NaNTrap(const G4Step* step ) const;
    
private:

  void AssignSD(const std::string & vname);

  std::vector<std::string> namesOfSD;
};

#endif
