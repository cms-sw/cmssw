// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelOrganization
//
// Implementation:
//     <Notes on implementation>
//


// system include files

// user include files
#include "SimG4CMS/PPS/interface/PPSPixelOrganization.h"
//#include "SimG4CMS/Forward/interface/TotemNumberMerger.h"
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

					      //				,	_currentLayer(-1),
					      //	_currentObjectType(Undefined) 
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

//  std::cout << " INSIDE PPSPixelOrganization :: GetUnitID " << std::endl;

//  _currentDetectorPosition = 0; // default value for TestBeam geometry

//  int currLAOT;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4VPhysicalVolume* physVol;
  int ii =0;
//  unsigned int Arm = 0;
// unsigned int Station = 0;
//  unsigned int RP = 0;
//  unsigned int Plane = 0;



//  G4String temp;


  for ( ii = 0; ii < touch->GetHistoryDepth(); ii++ ){
    physVol = touch->GetVolume(ii);


   edm::LogInfo("PPSSim")<< "physVol=" << physVol->GetName()
			   << ", level=" << ii  << ", physVol->GetCopyNo()=" 
			   << physVol->GetCopyNo()
      ;

/*
    if(physVol->GetName() == "RP_box_secondary_vacuum" && 
       physVol->GetCopyNo()==0) _currentArm = 0;
    if(physVol->GetName() == "RP_box_secondary_vacuum" && 
       physVol->GetCopyNo()==1) _currentArm = 1;
    std::cout<< " ***** WARNING TO BE FIXED: using  RP_box_secondary_vacuum to select Arm in PPSPixelOrganization.cc. " <<std::endl; 
*/



  if(physVol->GetName().contains("Envelop")){
//    _currentRP = 0; 
//    std::cout<< " **** WARNING TO BE FIXED: using _currentRP = 0 in PPSPixelOrganization.cc " <<std::endl; 

    _currentPlane = physVol->GetCopyNo()-1;
  }
    
     else if(physVol->GetName() == "RP_box_primary_vacuum")
    {
//      UNITA+=10*physVol->GetCopyNo();
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


