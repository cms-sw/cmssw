
#include "Validation/Geometry/interface/MaterialBudgetAction.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4UImanager.hh"
#include "G4UIterminal.hh"
#include "G4UItcsh.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4TouchableHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4TransportationManager.hh"

#include <iostream>


//-------------------------------------------------------------------------
MaterialBudgetAction::MaterialBudgetAction(const edm::ParameterSet& iPSet) :
  theHistoMgr(0)
{
  theData = new MaterialBudgetData;
  
  edm::ParameterSet m_Anal = iPSet.getParameter<edm::ParameterSet>("MaterialBudgetAction");
  
  //---- Accumulate material budget only inside selected volumes
  std::string theHistoList = m_Anal.getParameter<std::string>("HistogramList");
  std::vector<std::string> volList = m_Anal.getParameter< std::vector<std::string> >("SelectedVolumes");
  std::vector<std::string>::const_iterator ite;
  std::cout << "TestGeometry: List of the selected volumes: " << std::endl;
  for( ite = volList.begin(); ite != volList.end(); ite++ ){
    if( (*ite) != "None" ) {
      theVolumeList.push_back( *ite );
      std::cout << (*ite) << std::endl;
    }
  }
  // log
  if(theHistoList == "Tracker" ) {
    std::cout << "TestGeometry: MaterialBudgetAction running in Tracker Mode" << std::endl;
  } 
  else if(theHistoList == "ECAL" ) {
    std::cout << "TestGeometry: MaterialBudgetAction running in Ecal Mode" << std::endl;
  } 
  else {
    std::cout << "TestGeometry: MaterialBudgetAction running in General Mode" << std::endl;
  }
  //
    
  //---- Stop track when a process occurs
  theProcessToStop = m_Anal.getParameter<std::string>("StopAfterProcess");
  std::cout << "TestGeometry: stop at process " << theProcessToStop << std::endl;

  //---- Save histos to ROOT file 
  std::string saveToHistosFile = m_Anal.getParameter<std::string>("HistosFile");
  if( saveToHistosFile != "None" ) {
    saveToHistos = true;
    std::cout << "TestGeometry: saving histograms to " << saveToHistosFile << std::endl;
    theHistoMgr = new TestHistoMgr();

    // rr
    if(theHistoList == "Tracker" ) {
      theHistos = new MaterialBudgetTrackerHistos( theData, theHistoMgr, saveToHistosFile );
    } 
    else if (theHistoList == "ECAL") {
      theHistos = new MaterialBudgetEcalHistos( theData, theHistoMgr, saveToHistosFile );
    }
    else {
      theHistos = new MaterialBudgetHistos( theData, theHistoMgr, saveToHistosFile );
    }
      // rr
  } else {
    saveToHistos = false;
  }
  
  //---- Save material budget info to TEXT file
  std::string saveToTxtFile = m_Anal.getParameter<std::string>("TextFile");
  if( saveToTxtFile != "None" ) {
    saveToTxt = true;
    std::cout << "TestGeometry: saving text info to " << saveToTxtFile << std::endl;
    theTxt = new MaterialBudgetTxt( theData, saveToTxtFile );
  } else {
    saveToTxt = false;
  }
  
  //---- Compute all the steps even if not stored on file
  bool allSteps = m_Anal.getParameter<bool>("AllStepsToTree");  
  std::cout << "TestGeometry: all steps are computed " << allSteps << std::endl;
  if( allSteps ) theData->SetAllStepsToTree();
  
  //---- Save tree to ROOT file
  std::string saveToTreeFile = m_Anal.getParameter<std::string>("TreeFile");
  //  std::string saveToTreeFile = ""; 
  if( saveToTreeFile != "None" ) {
    saveToTree = true;
    theTree = new MaterialBudgetTree( theData, saveToTreeFile );
  } else {
    saveToTree = false;
  }
  std::cout << "TestGeometry: saving ROOT TREE to " << saveToTreeFile << std::endl;
  
  //---- Track the first decay products of the main particle
  // if their kinetic energy is greater than  Ekin
  storeDecay = m_Anal.getUntrackedParameter<bool>("storeDecay",false);  
  Ekin       = m_Anal.getUntrackedParameter<double>("EminDecayProd",1000.0); // MeV
  std::cout << "TestGeometry: decay products steps are stored " << storeDecay;
  if(storeDecay) std::cout << " if their kinetic energy is greater than " << Ekin << " MeV";
  std::cout << std::endl;
  firstParticle = false;
  }


//-------------------------------------------------------------------------
MaterialBudgetAction::~MaterialBudgetAction()
{
  if (saveToTxt) delete theTxt;
  if (saveToTree) delete theTree;
  if (saveToHistos) delete theHistos;
  if (theHistoMgr) delete theHistoMgr;
  delete theData;
}


//-------------------------------------------------------------------------
void MaterialBudgetAction::produce(edm::Event& e, const edm::EventSetup&)
{
}


//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const BeginOfRun* trk)
{
  //----- Check that selected volumes are indeed part of the geometry
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume*>::const_iterator lvcite;
  std::vector<G4String>::const_iterator volcite;

  for( volcite = theVolumeList.begin(); volcite != theVolumeList.end(); volcite++ ){
  //-  std::cout << " MaterialBudgetAction checking volume " << *volcite << std::endl;
    bool volFound = false;
    for( lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++ ) {
      if( (*lvcite)->GetName() == *volcite )  {
	volFound = true;
	break;
      }
    }
    if( !volFound ) {
      std::cerr << " @@@@@@@ WARNING at MaterialBudgetAction: selected volume not found in geometry " << *volcite << std::endl;
    }
  }


  //----- Check process selected is one of the available ones
  bool procFound = false;
  if( theProcessToStop == "None" ) { 
    procFound = true;
  } else {
    G4ParticleTable * partTable = G4ParticleTable::GetParticleTable();
    int siz = partTable->size();
    for (int ii= 0; ii < siz; ii++) {
      G4ParticleDefinition * particle = partTable->GetParticle(ii);
      std::string particleName = particle->GetParticleName();
      
      //--- All processes of this particle 
      G4ProcessManager * pmanager = particle->GetProcessManager();
      G4ProcessVector * pvect = pmanager->GetProcessList();
      int sizproc = pvect->size();
      for (int jj = 0; jj < sizproc; jj++) {
	if( (*pvect)[jj]->GetProcessName() == theProcessToStop ) {
	  procFound = true;
	  break;
	}
      }
    }
  }

  if( !procFound ) {
      std::cerr << " @@@@@@@ WARNING at MaterialBudgetAction: selected process to stop tracking not found " << theProcessToStop << std::endl;
    }

}


//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const BeginOfTrack* trk)
{
  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  
  // that was a temporary action while we're sorting out
  // about # of secondaries (produced if CutsPerRegion=true)
  //
  std::cout << "Track ID " << aTrack->GetTrackID() << " Track parent ID " << aTrack->GetParentID() 
	    << " PDG Id. = " << aTrack->GetDefinition()->GetPDGEncoding()
	    << " Ekin = " << aTrack->GetKineticEnergy() << " MeV" << std::endl;
  if( aTrack->GetCreatorProcess() ) std::cout << " produced through " << aTrack->GetCreatorProcess()->GetProcessType() << std::endl;
  
  if(aTrack->GetTrackID() == 1) {
    firstParticle = true;
  } else {
    firstParticle = false;
  }
  
  if( storeDecay ) { // if record of the decay is requested
    if( aTrack->GetCreatorProcess() ) {
      if (
	  aTrack->GetParentID() == 1
	  &&
	  //	  aTrack->GetCreatorProcess()->GetProcessType() == 6
	  //	  &&
	  aTrack->GetKineticEnergy() > Ekin
	  ) {
	// continue
      } else {
	G4Track * aTracknc = const_cast<G4Track*>(aTrack);
	aTracknc->SetTrackStatus(fStopAndKill);
	return;
      }
    } // particles produced from a decay (type=6) of the main particle (ID=1) with Kinetic Energy [MeV] > Ekin
  } else { // kill all the other particles (take only the main one until it disappears) if decay not stored
    if( aTrack->GetParentID() != 0) {
      G4Track * aTracknc = const_cast<G4Track*>(aTrack);
      aTracknc->SetTrackStatus(fStopAndKill);
      return;  
    }
  }
  
  
  if(firstParticle) {
    //--------- start of track
    //-    std::cout << " Data Start Track " << std::endl;
    theData->dataStartTrack( aTrack );
    if (saveToTree) theTree->fillStartTrack();
    if (saveToHistos) theHistos->fillStartTrack();
    if (saveToTxt) theTxt->fillStartTrack();
  }
}
 

//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const G4Step* aStep)
{

  //----- Check it is inside one of the volumes selected
  if( theVolumeList.size() != 0 ) {
    if( !CheckTouchableInSelectedVolumes( aStep->GetTrack()->GetTouchable() ) ) return;
  } 

  //---------- each step
  theData->dataPerStep( aStep );
  //-  std::cout << " aStep->GetPostStepPoint()->GetTouchable() " << aStep->GetPostStepPoint()->GetTouchable()->GetVolume() << " " << aStep->GetPreStepPoint()->GetTouchable()->GetVolume() << std::endl;
  if (saveToTree) theTree->fillPerStep();
  if (saveToHistos) theHistos->fillPerStep();
  if (saveToTxt) theTxt->fillPerStep();


  //----- Stop tracking after selected process
  if( StopAfterProcess( aStep ) ) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus( fStopAndKill );
  }

  return;

}


//-------------------------------------------------------------------------
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


//-------------------------------------------------------------------------
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



//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const EndOfTrack* trk)
{
  //  std::cout << " EndOfTrack " << saveToHistos << std::endl;
  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  //  if( aTrack->GetParentID() != 0 ) return;
  
  //---------- end of track (OutOfWorld)
  //-  std::cout << " Data End Track " << std::endl;
  theData->dataEndTrack( aTrack );
}

void MaterialBudgetAction::update(const EndOfEvent* evt)
{
  //-  std::cout << " Data End Event " << std::endl;
  if (saveToTree) theTree->fillEndTrack();
  if (saveToHistos) theHistos->fillEndTrack();
  if (saveToTxt) theTxt->fillEndTrack();  
}

//-------------------------------------------------------------------------
void MaterialBudgetAction::endRun()
{
}


//-------------------------------------------------------------------------
bool MaterialBudgetAction::CheckTouchableInSelectedVolumes( const G4VTouchable*  touch ) 
{
  std::vector<G4String>::const_iterator ite;
  size_t volh = touch->GetHistoryDepth();
  for( ite = theVolumeList.begin(); ite != theVolumeList.end(); ite++ ){
    //-  std::cout << " CheckTouchableInSelectedVolumes vol " << *ite << std::endl;
    for( int ii = volh; ii >= 0; ii-- ){
      //-  std::cout << ii << " CheckTouchableInSelectedVolumes parent  " << touch->GetVolume(ii)->GetName() << std::endl;
      if( touch->GetVolume(ii)->GetName() == *ite ) return true;
    }
  }

  return false;

}


//-------------------------------------------------------------------------
bool MaterialBudgetAction::StopAfterProcess( const G4Step* aStep )
{
  if( theProcessToStop == "" ) return false;

  if(aStep->GetPostStepPoint()->GetProcessDefinedStep() == NULL) return false;
  if( aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == theProcessToStop ) {
    std::cout << " MaterialBudgetAction::StopAfterProcess " << aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << std::endl;
    return true;
  } else {
    return false;
  }
}
