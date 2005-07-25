#ifndef SimG4Core_SensitiveDetector_H
#define SimG4Core_SensitiveDetector_H

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "G4VSensitiveDetector.hh"

#include <string>

class G4Step;
class G4HCofThisEvent;
class G4TouchableHistory;
class G4VPhysicalVolume;

class SensitiveDetector : public G4VSensitiveDetector
{
public:
    explicit SensitiveDetector(std::string & iname);
    virtual ~SensitiveDetector();
    virtual void Initialize(G4HCofThisEvent * eventHC);
    virtual void clearHits() = 0;
    virtual G4bool ProcessHits(G4Step * step ,G4TouchableHistory * tHistory) = 0;
    virtual int SetDetUnitId(G4Step * step) = 0;
    void Register();
    virtual void AssignSD(std::string & vname); 
    virtual void EndOfEvent(G4HCofThisEvent * eventHC); 
    enum coordinates {WorldCoordinates, LocalCoordinates};
    Local3DPoint InitialStepPosition(G4Step * s, coordinates);
    Local3DPoint FinalStepPosition(G4Step * s, coordinates);
    Local3DPoint ConvertToLocal3DPoint(G4ThreeVector point);
    
    std::string nameOfSD(){return name;}

    virtual std::vector<std::string> getNames() {
      std::vector<std::string> temp;
      temp.push_back(nameOfSD());
      return temp;
    }

private:
    std::string name;
    G4Step * currentStep;
};

#endif
