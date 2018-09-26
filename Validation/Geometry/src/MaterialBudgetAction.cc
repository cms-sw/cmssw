
#include "Validation/Geometry/interface/MaterialBudgetAction.h"
#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
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
MaterialBudgetAction::MaterialBudgetAction(const edm::ParameterSet& iPSet)
{
  theData = std::make_shared<MaterialBudgetData>();

  edm::ParameterSet m_Anal = iPSet.getParameter<edm::ParameterSet>("MaterialBudgetAction");
  
  //---- Accumulate material budget only inside selected volumes
  std::string theHistoList = m_Anal.getParameter<std::string>("HistogramList");
  std::vector<std::string> volList = m_Anal.getParameter< std::vector<std::string> >("SelectedVolumes");

  edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: List of the selected volumes:";
  for( const auto& it : volList) {
    if( it != "None" ) {
      theVolumeList.push_back(it);
      edm::LogInfo("MaterialBudget") << it ;
    }
  }

  // log
  if(theHistoList == "Tracker" ) {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: running in Tracker Mode";
  } 
  else if(theHistoList == "ECAL" ) {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: running in Ecal Mode";
  } 
  else if(theHistoList == "HGCal" ) {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: running in HGCal Mode";
  } 
  else {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: running in General Mode";
  }
    
  //---- Stop track when a process occurs
  theProcessToStop = m_Anal.getParameter<std::string>("StopAfterProcess");
  LogDebug("MaterialBudget") << "MaterialBudgetAction: stop at process " << theProcessToStop;

  //---- Save histos to ROOT file 
  std::string saveToHistosFile = m_Anal.getParameter<std::string>("HistosFile");
  if( saveToHistosFile != "None" ) {
    saveToHistos = true;
    edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: saving histograms to " << saveToHistosFile;
    theHistoMgr = std::make_shared<TestHistoMgr>();
    if(theHistoList == "Tracker" ) {
      theHistos = std::make_shared<MaterialBudgetTrackerHistos>(theData, theHistoMgr, saveToHistosFile);
    }
    else if (theHistoList == "ECAL") {
      theHistos = std::make_shared<MaterialBudgetEcalHistos>(theData, theHistoMgr, saveToHistosFile);
    }
    else if (theHistoList == "HGCal") {
      theHistos = std::make_shared<MaterialBudgetHGCalHistos>( theData, theHistoMgr, saveToHistosFile );
      //In HGCal mode, so tell data class
      theData->setHGCalmode(true);
    }
    else {
      theHistos = std::make_shared<MaterialBudgetHistos>( theData, theHistoMgr, saveToHistosFile ); 
    }
  } else {
    edm::LogWarning("MaterialBudget") << "MaterialBudgetAction: No histograms file specified";
    saveToHistos = false;
  }
  
  //---- Save material budget info to TEXT file
  std::string saveToTxtFile = m_Anal.getParameter<std::string>("TextFile");
  if( saveToTxtFile != "None" ) {
    saveToTxt = true;
    edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: saving text info to " << saveToTxtFile;
    theTxt = std::make_shared<MaterialBudgetTxt>( theData, saveToTxtFile );
  } else {
    saveToTxt = false;
  }
  
  //---- Compute all the steps even if not stored on file
  bool allSteps = m_Anal.getParameter<bool>("AllStepsToTree");  
  edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: all steps are computed " << allSteps;
  if( allSteps ) theData->SetAllStepsToTree();
  
  //---- Save tree to ROOT file
  std::string saveToTreeFile = m_Anal.getParameter<std::string>("TreeFile");
  if( saveToTreeFile != "None" ) {
    saveToTree = true;
    theTree = std::make_shared<MaterialBudgetTree>( theData, saveToTreeFile );
  } else {
    saveToTree = false;
  }
  edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: saving ROOT TTree to " << saveToTreeFile;
  
  //---- Track the first decay products of the main particle
  // if their kinetic energy is greater than  Ekin
  storeDecay = m_Anal.getUntrackedParameter<bool>("storeDecay",false);  
  Ekin       = m_Anal.getUntrackedParameter<double>("EminDecayProd",1000.0); // MeV
  edm::LogInfo("MaterialBudget") << "MaterialBudgetAction: decay products steps are stored (" 
				 << storeDecay << ") if their kinetic energy is greater than " 
				 << Ekin << " MeV";
  firstParticle = false;
}


//-------------------------------------------------------------------------
MaterialBudgetAction::~MaterialBudgetAction()
{
}

//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const BeginOfRun* )
{
  //----- Check that selected volumes are indeed part of the geometry
  const G4LogicalVolumeStore* lvs = G4LogicalVolumeStore::GetInstance();

  for(const auto& volcite: theVolumeList) {
    bool volFound = false;
    for(const auto& lvcite: *lvs) {
      if( lvcite->GetName() == volcite )  {
	volFound = true;
	break;
      }
    }
    if( !volFound ) {
       edm::LogWarning("MaterialBudget") << "MaterialBudgetAction: selected volume not found in geometry " << volcite;
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
    edm::LogWarning("MaterialBudget") << "MaterialBudgetAction: selected process to stop tracking not found " << theProcessToStop;
  }
}

//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const BeginOfTrack* trk)
{
  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  
  // that was a temporary action while we're sorting out
  // about # of secondaries (produced if CutsPerRegion=true)

  LogDebug("MaterialBudget") << "MaterialBudgetAction: Track ID " << aTrack->GetTrackID() 
				  << "Track parent ID " << aTrack->GetParentID() 
				  << "PDG Id. = " << aTrack->GetDefinition()->GetPDGEncoding()
				  << "Ekin = " << aTrack->GetKineticEnergy() << " MeV";

  if( aTrack->GetCreatorProcess() ) 
    LogDebug("MaterialBudget") << "MaterialBudgetAction: produced through " << aTrack->GetCreatorProcess()->GetProcessType();
  
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
  
  theData->dataStartTrack( aTrack );

  if (saveToTree) theTree->fillStartTrack();
  if (saveToHistos) theHistos->fillStartTrack();
  if (saveToTxt) theTxt->fillStartTrack();
}
 
//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const G4Step* aStep)
{
  //----- Check it is inside one of the volumes selected
  if( !theVolumeList.empty() ) {
    if( !CheckTouchableInSelectedVolumes( aStep->GetTrack()->GetTouchable() ) ) return;
  }

  //---------- each step
  theData->dataPerStep( aStep );
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
  const G4TouchableHistory* theTouchable
    = (const G4TouchableHistory*)(aStepPoint->GetTouchable());
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
  const G4TouchableHistory* theTouchable
    = (const G4TouchableHistory*)(aStepPoint->GetTouchable());
  G4int num_levels = theTouchable->GetHistoryDepth();
  
  if( theTouchable->GetVolume() ) {
    return theTouchable->GetVolume(num_levels-3)->GetName();
  } else { 
    return "OutOfWorld";
  }
}

//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const EndOfTrack* trk)
{
  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  theData->dataEndTrack( aTrack );

  if (saveToTree) theTree->fillEndTrack();
  if (saveToHistos) theHistos->fillEndTrack();
  if (saveToTxt) theTxt->fillEndTrack();
}

//-------------------------------------------------------------------------
void MaterialBudgetAction::update(const EndOfRun* )
{
  // endOfRun calls TestHistoMgr::save() allowing to write 
  // the ROOT files containing the histograms

  if (saveToHistos) theHistos->endOfRun();
  if (saveToTxt) theHistos->endOfRun();
  if (saveToTree) theTree->endOfRun();

  return;
}

//-------------------------------------------------------------------------
bool MaterialBudgetAction::CheckTouchableInSelectedVolumes( const G4VTouchable*  touch ) 
{
  size_t volh = touch->GetHistoryDepth();
    for( int ii = volh; ii >= 0; ii-- ){
      if ( 
        std::find(theVolumeList.begin(),
                  theVolumeList.end(),
                  touch->GetVolume(ii)->GetName()) != theVolumeList.end() )
          return true;
    }
  return false;
}

//-------------------------------------------------------------------------
bool MaterialBudgetAction::StopAfterProcess( const G4Step* aStep )
{
  if( theProcessToStop.empty() ) return false;

  if( aStep->GetPostStepPoint()->GetProcessDefinedStep() == nullptr) return false;
  if( aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() == theProcessToStop ) {
    edm::LogInfo("MaterialBudget" )<< "MaterialBudgetAction :" 
				   << aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
    return true;
  } else {
    return false;
  }
}
