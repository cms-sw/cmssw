#include "SimG4Core/Application/interface/ExceptionHandler.h"
#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "G4EventManager.hh"
#include "G4StateManager.hh"
#include "G4ios.hh"

using std::cout;
using std::endl;
using std::string;

ExceptionHandler::ExceptionHandler(RunManager* rm) 
  : fRunManager(rm),fRunManagerMT(0)
{}

ExceptionHandler::ExceptionHandler(RunManagerMT* rm) 
  : fRunManager(0),fRunManagerMT(rm)
{}

ExceptionHandler::~ExceptionHandler() {}

bool ExceptionHandler::Notify(const char* exceptionOrigin,
			      const char* exceptionCode,
			      G4ExceptionSeverity severity,
			      const char* description)
{
    cout << endl;
    cout << "*** G4Exception : " << exceptionCode << " issued by " << exceptionOrigin << endl;
    cout << "    " << description << endl;
    bool abortionForCoreDump = false;
    G4ApplicationState aps = G4StateManager::GetStateManager()->GetCurrentState();
    switch(severity)
    {
    case FatalException:
        if ( aps==G4State_EventProc && exceptionOrigin==string("G4HadronicProcess") )
        {
        cout << "*** Fatal exception *** " << endl;
        throw SimG4Exception( "SimG4CoreApplication: Bug in G4HadronicProcess" ) ;
        }
        else
        {
        cout << "*** Fatal exception *** core dump ***" << endl;
        abortionForCoreDump = true;
        }
/*
	cout << "*** Fatal exception *** core dump ***" << endl;
	abortionForCoreDump = true;
*/
	break;
    case FatalErrorInArgument:
	cout << "*** Fatal error in argument *** core dump ***" << endl;
	abortionForCoreDump = true;
	break;
    case RunMustBeAborted:
	if(aps==G4State_GeomClosed || aps==G4State_EventProc)
	{
	    cout << "*** Run must be aborted " << endl;
	    if(fRunManager) { fRunManager->abortRun(false); }
	    if(fRunManagerMT) { fRunManagerMT->abortRun(false); }
	}
	abortionForCoreDump = false;
	break;
    case EventMustBeAborted:
	if(aps==G4State_EventProc)
	{
	    cout << "*** Event must be aborted " << endl;
	    // if (override && exceptionCode == string("StuckTrack"))
/*
	    if (exceptionCode == string("StuckTrack"))
	    {
		//if (verbose > 1) cout << "*** overriden by user " << endl;
		G4Track * t = G4EventManager::GetEventManager()->GetTrackingManager()->GetTrack();
		//if (verbose > 1) 
		    cout << " ERROR - G4Navigator::ComputeStep() " << endl 
			 << " Track " << t->GetTrackID() << " stuck " 
			 << " in volume " << t->GetVolume()->GetName()
			 << " at point " << t->GetPosition()/mm << " mm "<< endl
			 << " with direction: " << t->GetMomentumDirection()
			 << " and distance to out " 
			 << (t->GetVolume()->GetLogicalVolume()->GetSolid())
			->DistanceToOut(t->GetPosition())/mm << " mm " << endl;
		//if (verbose > 1) 
		    cout << " Particle " << t->GetDynamicParticle()->GetDefinition()->GetParticleName()
			 << " from parent ID " << t->GetParentID() << endl
			 << " with " << t->GetKineticEnergy()/MeV << " MeV kinetic energy " 
			 << " created in " << t->GetLogicalVolumeAtVertex()->GetName() << endl;
		cout << " *** StuckTrack: track status set to fStopButAlive " << endl;
		t->SetTrackStatus(fStopButAlive);
	    }
	    else
*/
		//fRunManager->abortEvent();
		//RunManager::instance()->abortEvent() ;
		throw SimG4Exception( "SimG4CoreApplication: G4Navigator:StuckTrack detected" ) ;
	}
	abortionForCoreDump = false;
	break;
    default:
	cout << "*** This is just a warning message " << endl;
	abortionForCoreDump = false;
	break;
    }
    cout << endl;
    return abortionForCoreDump;
}

