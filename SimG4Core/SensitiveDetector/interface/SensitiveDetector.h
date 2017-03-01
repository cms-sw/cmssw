#ifndef SimG4Core_SensitiveDetector_H
#define SimG4Core_SensitiveDetector_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "G4VSensitiveDetector.hh"

#include <boost/cstdint.hpp>
#include <string>

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;
class G4VPhysicalVolume;
class DDCompactView;    

class SensitiveDetector : public G4VSensitiveDetector
{
public:
  explicit SensitiveDetector(std::string & iname, const DDCompactView & cpv,
			     const SensitiveDetectorCatalog & ,
			     edm::ParameterSet const & p);
  virtual ~SensitiveDetector();
  virtual void Initialize(G4HCofThisEvent * eventHC);
  virtual void clearHits() = 0;
  virtual G4bool ProcessHits(G4Step * step ,G4TouchableHistory * tHistory) = 0;
  virtual uint32_t setDetUnitId(G4Step * step) = 0;
  void Register();
  virtual void AssignSD(const std::string & vname);
  virtual void EndOfEvent(G4HCofThisEvent * eventHC); 
  enum coordinates {WorldCoordinates, LocalCoordinates};
  Local3DPoint InitialStepPosition(G4Step * s, coordinates);
  Local3DPoint FinalStepPosition(G4Step * s, coordinates);
  Local3DPoint ConvertToLocal3DPoint(const G4ThreeVector& point);    
  std::string nameOfSD() { return name; }
  virtual std::vector<std::string> getNames() 
  {
    std::vector<std::string> temp;
    temp.push_back(nameOfSD());
    return temp;
  }
  
  void NaNTrap( G4Step* step ) ;
    
private:
  std::string name;
  G4Step * currentStep;
};

#endif
