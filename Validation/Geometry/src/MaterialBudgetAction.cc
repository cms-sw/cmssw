
#include "Validation/Geometry/interface/MaterialBudgetAction.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "G4Step.hh"
#include "G4SDManager.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"

#include <iostream>

#include "G4Track.hh"

#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "G4UItcsh.hh"

#include "G4TransportationManager.hh"


MaterialBudgetAction::MaterialBudgetAction(const edm::ParameterSet& iPSet) 
{
  theData = new MaterialBudgetData;

  edm::ParameterSet m_Anal = iPSet.getParameter<edm::ParameterSet>("MaterialBudgetAction");
 
  //---- Save histos to ROOT file 
  std::string saveToHistosFile = m_Anal.getParameter<std::string>("HistosFile");
  if( saveToHistosFile != "" ) {
    saveToHistos = true;
    std::cout << "TestGeometry: saving histograms to " << saveToHistosFile << std::endl;
    theHistos = new MaterialBudgetHistos( theData, saveToHistosFile );
  } else {
    saveToHistos = false;
  }


  //---- Save material budget info to TEXT file
  std::string saveToTxtFile = m_Anal.getParameter<std::string>("TextFile");
  if( saveToTxtFile != "" ) {
    saveToTxt = true;
    std::cout << "TestGeometry: saving text info to " << saveToTxtFile << std::endl;
    theTxt = new MaterialBudgetTxt( theData, saveToTxtFile );
  } else {
    saveToTxt = false;
  }

  //---- Save tree to ROOT file
  std::string saveToTreeFile = m_Anal.getParameter<std::string>("TreeFile");
  //  std::string saveToTreeFile = ""; 
  if( saveToTreeFile != "" ) {
    saveToTree = true;
    std::cout << "TestGeometry: saving ROOT TREE to " << saveToTreeFile << std::endl;
    theTree = new MaterialBudgetTree( theData, saveToTreeFile );

    bool allSteps = m_Anal.getParameter<bool>("AllStepsToTree");  
    if( allSteps ) theData->SetAllStepsToTree();

  } else {
    saveToTree = false;
  }
  
  
}


MaterialBudgetAction::~MaterialBudgetAction()
{
  if (saveToTxt) delete theTxt;
  if (saveToTree) delete theTree;
  if (saveToHistos) delete theHistos;
  delete theData;
}

void
MaterialBudgetAction::produce(edm::Event& e, const edm::EventSetup&)
{
}

void MaterialBudgetAction::update(const BeginOfTrack* trk)
{
  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  //--------- start of track
  theData->dataStartTrack( aTrack );
  if (saveToTree) theTree->fillStartTrack();
  if (saveToHistos) theHistos->fillStartTrack();
  if (saveToTxt) theTxt->fillStartTrack();
}
 
void MaterialBudgetAction::update(const G4Step* aStep)
{

  //---------- each step
  theData->dataPerStep( aStep );
  //-  std::cout << " aStep->GetPostStepPoint()->GetTouchable() " << aStep->GetPostStepPoint()->GetTouchable()->GetVolume() << " " << aStep->GetPreStepPoint()->GetTouchable()->GetVolume() << std::endl;
  if (saveToTree) theTree->fillPerStep();
  if (saveToHistos) theHistos->fillPerStep();
  if (saveToTxt) theTxt->fillPerStep();

  return;

}


std::string MaterialBudgetAction::getSubDetectorName( G4StepPoint* aStepPoint )
{
  G4TouchableHistory* theTouchable
    = (G4TouchableHistory*)(aStepPoint->GetTouchable());
  G4int num_levels = theTouchable->GetHistoryDepth();
  
  if( theTouchable->GetVolume() ) {
    return theTouchable->GetVolume(num_levels-1)->GetName();
  } else { 
    return "OutOfWorld";
  }
}


std::string MaterialBudgetAction::getPartName( G4StepPoint* aStepPoint )
{
  G4TouchableHistory* theTouchable
    = (G4TouchableHistory*)(aStepPoint->GetTouchable());
  G4int num_levels = theTouchable->GetHistoryDepth();
  //  theTouchable->MoveUpHistory(num_levels-3);
  
  if( theTouchable->GetVolume() ) {
    return theTouchable->GetVolume(num_levels-3)->GetName();
  } else { 
    return "OutOfWorld";
  }
}



void MaterialBudgetAction::update(const EndOfTrack* trk)
{
  std::cout << " EndOfTrack " << saveToHistos << std::endl;
  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  //---------- end of track (OutOfWorld)
  theData->dataEndTrack( aTrack );
  if (saveToTree) theTree->fillEndTrack();
  if (saveToHistos) theHistos->fillEndTrack();
  if (saveToTxt) theTxt->fillEndTrack();
}

void MaterialBudgetAction::endRun()
{
}

