#ifndef SimG4Core_DDG4Builder_h
#define SimG4Core_DDG4Builder_h

#include "SimG4Core/Notification/interface/DDG4DispContainer.h"

#include "DetectorDescription/DDCore/interface/DDLogicalPart.h"
#include "DetectorDescription/DDCore/interface/DDMaterial.h"
#include "DetectorDescription/DDCore/interface/DDSolid.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"

#include <map>
#include <vector>
#include <string>
 
#include "boost/signals.hpp"

class DDPosData;
class DDG4SolidConverter;

class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;
class G4VSolid;

using namespace edm::eventsetup;
using edm::EventSetup;
using edm::Timestamp;

class DDG4Builder 
{
public:
    DDG4Builder(const EventSetup&);  
    ~DDG4Builder(); 
    G4LogicalVolume * BuildGeometry();
    static DDG4DispContainer * theVectorOfDDG4Dispatchables();
protected:    
    G4VSolid * convertSolid(const DDSolid & dSolid);    
    G4LogicalVolume * convertLV(const DDLogicalPart & dLogical);  
    G4Material * convertMaterial(const DDMaterial & dMaterial);
    int getInt(const std::string & s, const DDLogicalPart & dLogical);
    double getDouble(const std::string & s, const DDLogicalPart & dLogical);
protected:
    DDG4SolidConverter * solidConverter_;
    std::map<DDMaterial,G4Material*> mats_;
    std::map<DDSolid,G4VSolid*> sols_;    
    std::map<DDLogicalPart,G4LogicalVolume*> logs_;
private:
    const EventSetup& eventSetup;
    static DDG4DispContainer * theVectorOfDDG4Dispatchables_;
};

#endif
