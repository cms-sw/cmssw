///////////////////////////////////////////////////////////////////////////////
// File: TrackingVerboseAction.cc
// Creation: P.Arce  09/01
// Modifications: porting to CMSSW by M. Stavrianakou 22/03/06
// Description:
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/TrackingVerbose/interface/TrackingVerboseAction.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Application/interface/TrackingAction.h"

#include "G4Track.hh"
#include "G4Event.hh"
#include "G4ios.hh"
#include "G4TrackingManager.hh"
#include "G4EventManager.hh"

using std::cout;
using std::endl;

TrackingVerboseAction::TrackingVerboseAction(edm::ParameterSet const & p) 
    : UtilityAction(p)
{
    fDEBUG = p.getUntrackedParameter<bool>("DEBUG",false);
    fHighEtPhotons = p.getUntrackedParameter<bool>("CheckForHighEtPhotons",false);

    //----- Set which events are verbose
    fTVEventMin  = p.getUntrackedParameter<int>("EventMin",0);
    fTVEventMax  = p.getUntrackedParameter<int>("EventMax",int(1E10));
    fTVEventStep = p.getUntrackedParameter<int>("EventStep",1);

    //----- Set which tracks of those events are verbose
    fTVTrackMin  = p.getUntrackedParameter<int>("TrackMin",0);
    fTVTrackMax  = p.getUntrackedParameter<int>("TrackMax",int(1E10));
    fTVTrackStep = p.getUntrackedParameter<int>("TrackStep",1);

    //----- Set the verbosity level
    fVerboseLevel = p.getUntrackedParameter<int>("VerboseLevel",1);

    if (fDEBUG)
    cout << "TV: fTVTrackMin " << fTVTrackMin   << " fTVTrackMax "  <<  fTVTrackMax 
	 <<  " fTVTrackStep "  << fTVTrackStep  << " fTVEventMin "  << fTVEventMin 
	 << " fTVEventMax "    << fTVEventMax   << " fTVEventStep " << fTVEventStep 
	 << " fVerboseLevel "  << fVerboseLevel << endl;

    //----- Set verbosity off to start
    fTrackingVerboseON = 0;
    fTkVerbThisEventON = 0;

    theTrackingManager = 0;
    cout << " TrackingVerbose constructed " << endl;
}

TrackingVerboseAction::~TrackingVerboseAction() {}

void TrackingVerboseAction::update(const BeginOfRun * run)
{
    TrackingAction * ta = 
	dynamic_cast<TrackingAction*>(G4EventManager::GetEventManager()->GetUserTrackingAction());
    theTrackingManager = ta->getTrackManager();
}

void TrackingVerboseAction::update(const BeginOfEvent * evt)
{
    if (evt==0) return;
    const G4Event * anEvent = (*evt)();
    if (anEvent==0) return;

    //----------- Set /tracking/verbose for this event 
    int eventNo = anEvent->GetEventID();
    if (fDEBUG) cout << "TV: trackID: NEW EVENT " << eventNo << endl;

    fTkVerbThisEventON = false;
    //----- Check if event is in the selected range
    bool trackingVerboseThisEvent = false;
    if (eventNo >= fTVEventMin && eventNo <= fTVEventMax) 
    {
	if ((eventNo-fTVEventMin) % fTVEventStep == 0) fTkVerbThisEventON = true;
    }

    if (fDEBUG)
    cout << " TV: fTkVerbThisEventON " <<  fTkVerbThisEventON 
	 << " fTrackingVerboseON " << fTrackingVerboseON 
	 << " fTVEventMin " << fTVEventMin << " fTVEventMax " << fTVEventMax << endl;
    //----- check if verbosity has to be changed
    if (fTkVerbThisEventON && !fTrackingVerboseON) 
    {
	SetTrackingVerbose(fVerboseLevel);
	fTrackingVerboseON = 1;
	if (fDEBUG) cout << "TV: VERBOSEet1 " << eventNo << endl;
    } 
    else if (trackingVerboseThisEvent == 0 && fTrackingVerboseON == 1) 
    {
	SetTrackingVerbose(0);
	fTrackingVerboseON = 0;
	if (fDEBUG) cout << "TV: VERBOSEet0 " << eventNo << endl;
    }

}

void TrackingVerboseAction::update(const BeginOfTrack * trk)
{
    const G4Track * aTrack = (*trk)();

    //----- High ET photon printout
    TrackInformation* trkInfo = dynamic_cast<TrackInformation*>   (aTrack->GetUserInformation());
    //---------- Set /tracking/verbose
    //----- track is verbose only if event is verbose
    double tkP = aTrack->GetMomentum().mag();
    double tkPx = aTrack->GetMomentum().x();
    double tkPy = aTrack->GetMomentum().y();
    double tkPz = aTrack->GetMomentum().z();

    double tvtx = aTrack->GetVertexPosition().x();
    double tvty = aTrack->GetVertexPosition().y();
    double tvtz = aTrack->GetVertexPosition().z();

    double g4t_phi=atan2(tkPy,tkPx);

    double drpart=sqrt(tkPx*tkPx + tkPy*tkPy);

    double mythetapart=acos(tkPz/sqrt(drpart*drpart+tkPz*tkPz));

    double g4t_eta=-log(tan(mythetapart/2.));
    G4int MytrackNo = aTrack->GetTrackID();
    
    if (fHighEtPhotons)
    {
	if (aTrack->GetDefinition()->GetParticleName() == "gamma" && aTrack->GetParentID() !=0)
	{
	    if((tkPx*tkPx + tkPy*tkPy + tkPz*tkPz)>1000.0*1000.0 &&
	       aTrack->GetCreatorProcess()->GetProcessName() == "LCapture")
	    {
		cout << "MY NEW GAMMA " << endl;
		cout << "**********************************************************************"  << endl;
		cout << "MY NEW TRACK ID = " << MytrackNo << "("
		     << aTrack->GetDefinition()->GetParticleName()
		     <<")"<< " PARENT ="<< aTrack->GetParentID() << endl;
		cout << "Primary particle: " 
		     << aTrack->GetDynamicParticle()->GetPrimaryParticle() << endl;
		cout << "Process type: " << aTrack->GetCreatorProcess()->GetProcessType()
		     << " Process name: " 
		     << aTrack->GetCreatorProcess()->GetProcessName() << endl;
		cout << "ToT E = " << aTrack->GetTotalEnergy() 
		     << " KineE = " << aTrack->GetKineticEnergy()
		     << " Tot P = " << tkP << " Pt = " << sqrt(tkPx*tkPx + tkPy*tkPy) 
		     << " VTX=(" << tvtx << "," << tvty << "," << tvtz << ")" << endl;
		if (aTrack->GetKineticEnergy() > 1.*GeV 
		    && aTrack->GetCreatorProcess()->GetProcessName() != "LCapture")
                cout << " KineE > 1 GeV !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
		const G4VTouchable* touchable=aTrack->GetTouchable();
		if (touchable!=0 && touchable->GetVolume()!=0 
		    && touchable->GetVolume()->GetLogicalVolume()!=0)
		{
		    G4Material* material=touchable->GetVolume()->GetLogicalVolume()->GetMaterial();
		    cout << "G4LCapture Gamma E(GeV) " 
			 << aTrack->GetTotalEnergy()/GeV << "  "
			 << material->GetName()<< " " 
			 << touchable->GetVolume()->GetName() << endl;
		    cout << "G4LCapture Gamma position(m): " 
			 << aTrack->GetPosition()/m << endl;
		    cout << "G4LCapture created Gamma direction " 
			 << aTrack->GetMomentumDirection() << endl;
		    cout << "G4LCapture gamma (eta,phi) = " 
			 << "(" << g4t_eta << "," << g4t_phi << ")" << endl;
		}
		aTrack->GetUserInformation()->Print();
		cout << "**********************************************************************"  << endl;
	    }
	}

	if (aTrack->GetDefinition()->GetParticleName() == "gamma")
	{
	    const G4VProcess * proc = aTrack->GetCreatorProcess();
	    double Tgamma = aTrack->GetKineticEnergy();
	    std::string ProcName;
	    const  std::string nullstr ("Null_prc");
	    if (proc) ProcName = proc->GetProcessName();
	    else ProcName = nullstr;
	    if (Tgamma > 2.5*GeV ) //&& ProcName!="Decay" && ProcName!="eBrem")
	    {
		std::string volumeName("_Unknown_Vol_");
		std::string materialName("_Unknown_Mat_");
		G4Material * material = 0;
		G4VPhysicalVolume * pvolume = 0;
		G4LogicalVolume * lvolume = 0;
		const G4VTouchable * touchable = aTrack->GetTouchable();
		if (touchable) pvolume = touchable->GetVolume();
		if (pvolume)
		{
		    volumeName = pvolume->GetName();
		    lvolume = pvolume->GetLogicalVolume();
		}
		if (lvolume) material = lvolume->GetMaterial();
		if (material) materialName = material->GetName();
		cout << "**** ALL photons > 2.5 GeV ****" << endl;
		cout << ProcName << "**** ALL photons: gamma E(GeV) "
		     << aTrack->GetTotalEnergy()/GeV << "  "
		     <<  materialName << " " << volumeName << endl;
		cout << ProcName << "**** ALL photons: gamma position(m): " 
		     << aTrack->GetPosition()/m << endl;
		cout << ProcName << "**** ALL photons: gamma direction " 
		     << aTrack->GetMomentumDirection() << endl;
		cout << "**********************************************************************"  << endl;
	    }
	}                                               
    }
    
    //---------- Set /tracking/verbose
    //----- track is verbose only if event is verbose
    if (fTkVerbThisEventON) 
    {
	int trackNo = aTrack->GetTrackID();    
	bool trackingVerboseThisTrack = false;
	//----- Check if track is in the selected range
	if (trackNo >= fTVTrackMin && trackNo <= fTVTrackMax) 
	{
	    if ((trackNo-fTVTrackMin) % fTVTrackStep == 0) trackingVerboseThisTrack = true;
	}
    
	//----- Set the /tracking/verbose for this track 
    if (trackingVerboseThisTrack == 1 && fTrackingVerboseON == 0) 
    {
	SetTrackingVerbose(fVerboseLevel);
	fTrackingVerboseON = 1;
	if (fDEBUG) cout << "TV: VERBOSEtt1 " << trackNo << endl;
    } 
    else if (!trackingVerboseThisTrack && ( fTrackingVerboseON )) 
    {
	SetTrackingVerbose(0);
	fTrackingVerboseON = 0;
	if (fDEBUG) cout << "TV: VERBOSEtt0 " << trackNo << endl;
    }
  }
}

void TrackingVerboseAction::SetTrackingVerbose(int verblev)
{
    if (fDEBUG) cout << " setting verbose level " << verblev << endl;
    if (theTrackingManager!=0) theTrackingManager->SetVerboseLevel(verblev);
}
 
