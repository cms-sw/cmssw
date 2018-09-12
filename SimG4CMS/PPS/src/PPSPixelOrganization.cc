// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelOrganization
//
// Implementation:
//     <Notes on implementation>
//

// user include files
#include "SimG4CMS/PPS/interface/PPSPixelOrganization.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh" 

//
// constructors and destructor
//
PPSPixelOrganization :: PPSPixelOrganization() : // _needUpdateUnitID(false),
    //	_needUpdateData(false),
    _currentUnitID(-1),
    _currentArm(-1),
    _currentStation(-1),
    _currentRP(-1),
    _currentPlane(-1)
{
    edm::LogInfo("PPSSim") << "Creating PPSPixelOrganization";
}

PPSPixelOrganization :: ~PPSPixelOrganization() {
}

//
// member functions
//

uint32_t PPSPixelOrganization :: GetUnitID(const G4Step* aStep) const {
    return const_cast<PPSPixelOrganization *>(this)->GetUnitID(aStep);
}


uint32_t PPSPixelOrganization :: GetUnitID(const G4Step* aStep) {

    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    G4VPhysicalVolume* physVol;
    int ii =0;

    for ( ii = 0; ii < touch->GetHistoryDepth(); ii++ ){
        physVol = touch->GetVolume(ii);

        edm::LogInfo("PPSSim")<< "physVol=" << physVol->GetName()
            << ", level=" << ii  << ", physVol->GetCopyNo()=" 
            << physVol->GetCopyNo()
            ;

        if(physVol->GetName().contains("Envelop")){
            _currentPlane = physVol->GetCopyNo()-1;
        }
        else if(physVol->GetName() == "RP_box_primary_vacuum")
        {
            int cpy_no = physVol->GetCopyNo();
            _currentArm = (cpy_no/100)%10;
            _currentStation = (cpy_no/10)%10;
            _currentRP = cpy_no%10;
        }

    }

    edm::LogInfo("PPSSim") <<_currentArm << " " << _currentRP << " " << _currentPlane;

    CTPPSPixelDetId id(_currentArm,_currentStation,_currentRP,_currentPlane);
    uint32_t kk = id.rawId();
    edm::LogInfo("PPSSim") << " ID " << kk;
    return id.rawId();

}


