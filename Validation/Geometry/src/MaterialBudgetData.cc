//#include "Utilities/UI/interface/SimpleConfigurable.h"

#include "Validation/Geometry/interface/MaterialBudgetData.h"
#include "G4Step.hh"
#include "G4Material.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

MaterialBudgetData::MaterialBudgetData()
{

  //instantiate categorizer to assing an ID to volumes and materials
  myMaterialBudgetCategorizer = 0;
}

void MaterialBudgetData::SetAllStepsToTree()
{
  allStepsToTree = 1;
  MAXNUMBERSTEPS = 0;
  MAXNUMBERSTEPS = 10000; //!!!WARNING: this number is also hardcoded when booking the tree
  theDmb = new float[MAXNUMBERSTEPS];
  theX = new float[MAXNUMBERSTEPS];
  theY = new float[MAXNUMBERSTEPS];
  theZ = new float[MAXNUMBERSTEPS];
  theVoluId = new int[MAXNUMBERSTEPS];
  theMateId = new int[MAXNUMBERSTEPS];

}


void MaterialBudgetData::dataStartTrack( const G4Track* aTrack )
{
  const G4ThreeVector& dir = aTrack->GetMomentumDirection() ;

  if( myMaterialBudgetCategorizer == 0) myMaterialBudgetCategorizer = new MaterialBudgetCategorizer;

  theStepN=0;
  theTotalMB=0;
  theEta=0;
  thePhi=0;

  if( dir.theta() != 0 ) {
    theEta = dir.eta(); 
  } else {
    theEta = -99;
  }
  thePhi = dir.phi()/deg;

}


void MaterialBudgetData::dataEndTrack( const G4Track* aTrack )
{
  std::cout << "[OVAL] MaterialBudget " << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() << " " << theEta << " " << thePhi << " " << theTotalMB << std::endl;
}


void MaterialBudgetData::dataPerStep( const G4Step* aStep )
{
  G4Material * theMaterialPre = aStep->GetPreStepPoint()->GetMaterial();
  G4Material * theMaterialPost = aStep->GetPostStepPoint()->GetMaterial();

  Hep3Vector prePos =  aStep->GetPreStepPoint()->GetPosition();
  Hep3Vector postPos =  aStep->GetPostStepPoint()->GetPosition();

  G4double steplen = aStep->GetStepLength();

  G4double radlen;

  radlen = theMaterialPre->GetRadlen();
  //t    radlen = theMaterialPre->GetNuclearInterLength();

  G4String name = theMaterialPre->GetName();
  //  std::cout << " steplen " << steplen << " radlen " << radlen << " mb " << steplen/radlen << " mate " << theMaterialPre->GetName() << std::endl;
     
  G4LogicalVolume* lv = aStep->GetTrack()->GetVolume()->GetLogicalVolume();

  // instantiate the categorizer
  int voluID = myMaterialBudgetCategorizer->volume( lv->GetName() );
  int mateID = myMaterialBudgetCategorizer->material( lv->GetMaterial()->GetName() );
  
  float dmb = steplen/radlen;
  Hep3Vector pos = aStep->GetPreStepPoint()->GetPosition();

  G4VPhysicalVolume* pv = aStep->GetTrack()->GetVolume();
  
  //fill data per step
  if( allStepsToTree ){
    theDmb[theStepN] = dmb; 
    if( stepN > MAXNUMBERSTEPS ) stepN = MAXNUMBERSTEPS;
    theX[theStepN] = pos.x();
    theY[theStepN] = pos.y();
    theZ[theStepN] = pos.z();
    theVoluId[theStepN] = voluID;
    theMateId[theStepN] = mateID;
  }

  theTrkLen = aStep->GetTrack()->GetTrackLength();
  //-  std::cout << " theTrkLen " << theTrkLen << " theTrkLen2 " << theTrkLen2 << " postPos " << postPos.mag() << postPos << std::endl;
  thePVname = pv->GetName();
  thePVcopyNo = pv->GetCopyNo();
  theRadLen = radlen;
  theTotalMB += dmb;
  theStepN++;
}

