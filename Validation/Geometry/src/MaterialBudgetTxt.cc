#include "Validation/Geometry/interface/MaterialBudgetTxt.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"
#include "G4EventManager.hh"
#include "G4Event.hh"


MaterialBudgetTxt::MaterialBudgetTxt( std::shared_ptr<MaterialBudgetData> data, const std::string& fileName )
  : MaterialBudgetFormat( data )
{
  const char * fnamechar = fileName.c_str();
  theFile = new std::ofstream(fnamechar, std::ios::out);
  edm::LogInfo("MaterialBudget") << "MaterialBudgetTxt: Dumping  Material Budget to " << fileName;
  if (theFile->fail()){
    edm::LogError("MaterialBudget") <<"MaterialBudgetTxt: Error opening file" << fileName;
  }
}


MaterialBudgetTxt::~MaterialBudgetTxt()
{
}


void MaterialBudgetTxt::fillStartTrack()
{
  
  (*theFile)<< " Track "<< G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() << " " << theData->getEta() << " " << theData->getPhi() << std::endl;
  // + 1 was GEANT3 notation  (*theFile)<< " Track "<< G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() + 1<< " " << theData->getEta() << " " << theData->getPhi() << std::endl;
}


void MaterialBudgetTxt::fillPerStep()
{
  (*theFile) << "step "<< theData->getTrkLen() << " " << theData->getPVname() << " " << theData->getPVcopyNo()  << " " << theData->getTotalMB() << " " << theData->getRadLen() << std::endl;
  //-    std::cout << "step "<< theData->getTrkLen() << " " << theData->getPVname() << " " << theData->getPVcopyNo()  << " " << theData->getTotalMB() << " " << theData->getRadLen() << std::endl;

}


void MaterialBudgetTxt::fillEndTrack()
{
  (*theFile) << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() << " " << "finalTrkMB " << theData->getTotalMB() << std::endl;
}

void MaterialBudgetTxt::endOfRun() 
{
  theFile->close();
}
