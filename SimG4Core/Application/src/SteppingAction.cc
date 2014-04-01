#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"

#include "G4LogicalVolumeStore.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "G4Track.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SteppingAction::SteppingAction(EventAction* e, const edm::ParameterSet & p) 
  : eventAction_(e), tracker(0), calo(0), initialized(false) {

  killBeamPipe = (p.getParameter<bool>("KillBeamPipe"));
  theCriticalEnergyForVacuum = (p.getParameter<double>("CriticalEnergyForVacuum")*CLHEP::MeV);
  theCriticalDensity = (p.getParameter<double>("CriticalDensity")*CLHEP::g/CLHEP::cm3);
  maxTrackTime  = (p.getParameter<double>("MaxTrackTime")*CLHEP::ns);
  maxTrackTimes = (p.getParameter<std::vector<double> >("MaxTrackTimes"));
  maxTimeNames  = (p.getParameter<std::vector<std::string> >("MaxTimeNames"));
  ekinMins      = (p.getParameter<std::vector<double> >("EkinThresholds"));
  ekinNames     = (p.getParameter<std::vector<std::string> >("EkinNames"));
  ekinParticles = (p.getParameter<std::vector<std::string> >("EkinParticles"));
  verbose       = (p.getUntrackedParameter<int>("Verbosity",0));

  edm::LogInfo("SimG4CoreApplication") << "SteppingAction:: KillBeamPipe = "
				       << killBeamPipe << " CriticalDensity = "
				       << theCriticalDensity*CLHEP::cm3/CLHEP::g << " g/cm3;"
				       << " CriticalEnergyForVacuum = "
				       << theCriticalEnergyForVacuum/CLHEP::MeV << " Mev;"
				       << " MaxTrackTime = " << maxTrackTime/CLHEP::ns 
				       << " ns";

  for (unsigned int i=0; i<maxTrackTimes.size(); i++) {
    maxTrackTimes[i] *= ns;
    edm::LogInfo("SimG4CoreApplication") << "SteppingAction::MaxTrackTime for "
					 << maxTimeNames[i] << " is " 
					 << maxTrackTimes[i];
  }

  killByTimeAtRegion = false;
  if(maxTrackTimes.size() > 0) { killByTimeAtRegion = true; }

  killByEnergy = false;
  if(ekinVolumes.size() > 0) {

    killByEnergy = true;
    edm::LogInfo("SimG4CoreApplication") << "SteppingAction::Kill following "
					 << ekinParticles.size() 
					 << " particles in " << ekinNames.size()
					 << " volumes";
    for (unsigned int i=0; i<ekinParticles.size(); i++) {
      ekinMins[i] *= GeV;
      edm::LogInfo("SimG4CoreApplication") << "SteppingAction::Particle " << i
					   << " " << ekinParticles[i]
					   << " (Threshold = " << ekinMins[i]
					   << " MeV)";
    }
    for (unsigned int i=0; i<ekinNames.size(); i++) 
      edm::LogInfo("SimG4CoreApplication") << "SteppingAction::Volume[" << i
					   << "] = " << ekinNames[i];
  }
}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step * aStep) 
{
  if (!initialized) { initialized = initPointer(); }
  m_g4StepSignal(aStep);

  G4Track * theTrack = aStep->GetTrack();
  bool ok = (theTrack->GetTrackStatus() == fAlive);
  G4double kinEnergy = theTrack->GetKineticEnergy();

  if (ok && killBeamPipe && kinEnergy < theCriticalEnergyForVacuum
      && theTrack->GetDefinition()->GetPDGCharge() != 0.0 && kinEnergy > 0.0) {
    ok = catchLowEnergyInVacuum(theTrack, kinEnergy);
  }

  if(ok && aStep->GetPostStepPoint()->GetPhysicalVolume() != 0) {

    ok = catchLongLived(aStep); 

    if(ok && killByEnergy) { ok = killLowEnergy(aStep); }

    if(ok) {

      G4StepPoint* preStep = aStep->GetPreStepPoint();
      if(isThisVolume(preStep->GetTouchable(),tracker) &&
	 isThisVolume(aStep->GetPostStepPoint()->GetTouchable(),calo)) {

	math::XYZVectorD pos((preStep->GetPosition()).x(),
			     (preStep->GetPosition()).y(),
			     (preStep->GetPosition()).z());
      
	math::XYZTLorentzVectorD mom((preStep->GetMomentum()).x(),
				     (preStep->GetMomentum()).y(),
				     (preStep->GetMomentum()).z(),
				     preStep->GetTotalEnergy());
      
	uint32_t id = aStep->GetTrack()->GetTrackID();
      
	std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> p(pos,mom);
	eventAction_->addTkCaloStateInfo(id,p);
      }
    }
  }
}

bool SteppingAction::catchLowEnergyInVacuum(G4Track * theTrack, double theKenergy) 
{
  bool alive = true;
  if (theTrack->GetVolume()!=0) {
    double density = theTrack->GetVolume()->GetLogicalVolume()->GetMaterial()->GetDensity();
    if (density <= theCriticalDensity) {
      if (verbose>1) {
	edm::LogInfo("SimG4CoreApplication") 
	  <<   " SteppingAction: LoopCatchSteppingAction:catchLowEnergyInVacuumHere: "
	  << " Track from " << theTrack->GetDefinition()->GetParticleName()
	  << " of kinetic energy " << theKenergy/MeV << " MeV "
	  << " killed in " << theTrack->GetVolume()->GetLogicalVolume()->GetName()
	  << " of density " << density/(g/cm3) << " g/cm3" ;
      }
      theTrack->SetTrackStatus(fStopAndKill);
      alive = false;
    }
    if(alive) {
      density = theTrack->GetNextVolume()->GetLogicalVolume()->GetMaterial()->GetDensity();
      if (density <= theCriticalDensity) {
	if (verbose>1) {
	  edm::LogInfo("SimG4CoreApplication") 
	    << " SteppingAction: LoopCatchSteppingAction::catchLowEnergyInVacuumNext: "
	    << " Track from " << theTrack->GetDefinition()->GetParticleName()
	    << " of kinetic energy " << theKenergy/MeV << " MeV "
	    << " stopped in " << theTrack->GetVolume()->GetLogicalVolume()->GetName()
	    << " before going into "<< theTrack->GetNextVolume()->GetLogicalVolume()->GetName()
	    << " of density " << density/(g/cm3) << " g/cm3" ;
	}
	theTrack->SetTrackStatus(fStopButAlive);
	alive = false;
      }
    }
  }
  return alive;
}

bool SteppingAction::catchLongLived(const G4Step * aStep) 
{
  bool flag   = true;
  double time = (aStep->GetPostStepPoint()->GetGlobalTime())/nanosecond;
  double tofM = maxTrackTime;

  if(killByTimeAtRegion) {
    G4Region* reg = 
      aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion();
    for (unsigned int i=0; i<maxTimeRegions.size(); i++) {
      if (reg == maxTimeRegions[i]) {
	tofM = maxTrackTimes[i];
	break;
      }
    }
  }
  if (time > tofM) {
    killTrack(aStep);
    flag = false;
  }
  return flag;
}

bool SteppingAction::killLowEnergy(const G4Step * aStep) 
{
  bool ok = true;
  bool flag = false;
  G4LogicalVolume* lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  for (unsigned int i=0; i<ekinVolumes.size(); i++) {
    if (lv == ekinVolumes[i]) {
      flag = true;
      break;
    }
  }
  if (flag) {
    G4Track * track = aStep->GetTrack();
    double    ekin  = track->GetKineticEnergy();
    double    ekinM = 0;
    int       pCode = track->GetDefinition()->GetPDGEncoding();
    for (unsigned int i=0; i<ekinPDG.size(); i++) {
      if (pCode == ekinPDG[i]) {
	ekinM = ekinMins[i];
	break;
      }
    }
    if (ekin < ekinM) {
      killTrack(aStep);
      ok = false;
    }
  }
  return ok;
}
  
bool SteppingAction::initPointer() 
{
  bool flag = true;
  const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
  if (pvs) {
    std::vector<G4VPhysicalVolume*>::const_iterator pvcite;
    for (pvcite = pvs->begin(); pvcite != pvs->end(); pvcite++) {
      if ((*pvcite)->GetName() == "Tracker") tracker = (*pvcite);
      if ((*pvcite)->GetName() == "CALO")    calo    = (*pvcite);
      if (tracker && calo) break;
    }
    edm::LogInfo("SimG4CoreApplication") << "Pointer for Tracker " << tracker
					 << " and for Calo " << calo;
    if (tracker) LogDebug("SimG4CoreApplication") << "Tracker vol name "
						  << tracker->GetName();
    if (calo)    LogDebug("SimG4CoreApplication") << "Calorimeter vol name "
						  << calo->GetName();
  } else {
    flag = false;
  }

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int num = ekinNames.size();
  if (num > 0) {
    if (lvs) {
      std::vector<G4LogicalVolume*>::const_iterator lvcite;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
	for (unsigned int i=0; i<num; i++) {
	  if ((*lvcite)->GetName() == (G4String)(ekinNames[i])) {
	    ekinVolumes.push_back(*lvcite);
	    break;
	  }
	}
	if (ekinVolumes.size() == num) break;
      }
    }
    if (ekinVolumes.size() != num) flag = false;
    for (unsigned int i=0; i<ekinVolumes.size(); i++) {
      edm::LogInfo("SimG4CoreApplication") << ekinVolumes[i]->GetName()
					   <<" with pointer " <<ekinVolumes[i];
    }
  }

  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  for (unsigned int i=0; i<ekinParticles.size(); i++) {
    int pdg = theParticleTable->FindParticle(particleName=ekinParticles[i])->GetPDGEncoding();
    ekinPDG.push_back(pdg);
    edm::LogInfo("SimG4CoreApplication") << "Particle " << ekinParticles[i]
					 << " with code " << ekinPDG[i]
					 << " and KE cut off " << ekinMins[i];
  }
  if (!flag) edm::LogInfo("SimG4CoreApplication") << "SteppingAction fails to"
						  << " initialize some the "
						  << "LV pointers correctly";

  const G4RegionStore * rs = G4RegionStore::GetInstance();
  num = maxTimeNames.size();
  if (num > 0) {
    std::vector<double> tofs;
    if (rs) {
      std::vector<G4Region*>::const_iterator rcite;
      for (rcite = rs->begin(); rcite != rs->end(); rcite++) {
	for (unsigned int i=0; i<num; i++) {
	  if ((*rcite)->GetName() == (G4String)(maxTimeNames[i])) {
	    maxTimeRegions.push_back(*rcite);
	    tofs.push_back(maxTrackTimes[i]);
	    break;
	  }
	}
	if (tofs.size() == num) break;
      }
    }
    for (unsigned int i=0; i<tofs.size(); i++) {
      maxTrackTimes[i] = tofs[i];
      G4String name = "Unknown";
      if (maxTimeRegions[i]) name = maxTimeRegions[i]->GetName();
      edm::LogInfo("SimG4CoreApplication") << name << " with pointer " 
					   << maxTimeRegions[i]<<" KE cut off "
					   << maxTrackTimes[i];
    }
    if (tofs.size() != num) 
      edm::LogInfo("SimG4CoreApplication") << "SteppingAction fails to "
					   << "initialize some the region "
					   << "pointers correctly";
  }
  return true;
}

bool SteppingAction::isThisVolume(const G4VTouchable* touch, 
				  G4VPhysicalVolume* pv) 
{

  int level = ((touch->GetHistoryDepth())+1);
  if (level > 0 && level >= 3) {
    unsigned int ii = (unsigned int)(level - 3);
    return (touch->GetVolume(ii) == pv);
  }
  return false;
}

void SteppingAction::killTrack(const G4Step * aStep) 
{
  
  aStep->GetTrack()->SetTrackStatus(fStopAndKill);
  G4TrackVector tv = *(aStep->GetSecondary());
  for (unsigned int kk=0; kk<tv.size(); kk++) {
    if (tv[kk]->GetVolume() == aStep->GetPreStepPoint()->GetPhysicalVolume())
      tv[kk]->SetTrackStatus(fStopAndKill);
  }
  LogDebug("SimG4CoreApplication") 
    << "SteppingAction: Kills track " 
    << aStep->GetTrack()->GetTrackID() << " ("
    << aStep->GetTrack()->GetDefinition()->GetParticleName()
    << ") at " << aStep->GetPostStepPoint()->GetGlobalTime()/nanosecond << " ns in " 
    << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName()
    << " from region " 
    << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()->GetName();
}
