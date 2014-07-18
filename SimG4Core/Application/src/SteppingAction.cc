
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"

#include "G4LogicalVolumeStore.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog

SteppingAction::SteppingAction(EventAction* e, const edm::ParameterSet & p) 
  : eventAction_(e), tracker(0), calo(0), initialized(false), killBeamPipe(false) 
{
  theCriticalEnergyForVacuum = 
    (p.getParameter<double>("CriticalEnergyForVacuum")*CLHEP::MeV);
  if(0.0 < theCriticalEnergyForVacuum) { killBeamPipe = true; } 
  theCriticalDensity = 
    (p.getParameter<double>("CriticalDensity")*CLHEP::g/CLHEP::cm3);
  maxTrackTime    = p.getParameter<double>("MaxTrackTime")*CLHEP::ns;
  maxTrackTimes   = p.getParameter<std::vector<double> >("MaxTrackTimes");
  maxTimeNames    = p.getParameter<std::vector<std::string> >("MaxTimeNames");
  deadRegionNames = p.getParameter<std::vector<std::string> >("DeadRegions");
  ekinMins        = p.getParameter<std::vector<double> >("EkinThresholds");
  ekinNames       = p.getParameter<std::vector<std::string> >("EkinNames");
  ekinParticles   = p.getParameter<std::vector<std::string> >("EkinParticles");

  edm::LogInfo("SimG4CoreApplication") << "SteppingAction:: KillBeamPipe = "
				       << killBeamPipe << " CriticalDensity = "
				       << theCriticalDensity*CLHEP::cm3/CLHEP::g 
				       << " g/cm3;"
				       << " CriticalEnergyForVacuum = "
				       << theCriticalEnergyForVacuum/CLHEP::MeV 
				       << " Mev;"
				       << " MaxTrackTime = " 
				       << maxTrackTime/CLHEP::ns 
				       << " ns";

  numberTimes = maxTrackTimes.size();
  if(numberTimes > 0) {
    for (unsigned int i=0; i<numberTimes; i++) {
      edm::LogInfo("SimG4CoreApplication") << "SteppingAction::MaxTrackTime for "
					   << maxTimeNames[i] << " is " 
					   << maxTrackTimes[i] << " ns ";
      maxTrackTimes[i] *= ns;
    }
  }

  ndeadRegions =  deadRegionNames.size();
  if(ndeadRegions > 0) {
    edm::LogInfo("SimG4CoreApplication") 
      << "SteppingAction: Number of DeadRegions where all trackes are killed "
      << ndeadRegions;
    for(unsigned int i=0; i<ndeadRegions; ++i) {
      edm::LogInfo("SimG4CoreApplication") 
	<< "SteppingAction: DeadRegion " << i << ".  " << deadRegionNames[i];
    }
  }
  numberEkins = ekinNames.size();
  numberPart  = ekinParticles.size();
  if(0 == numberPart) { numberEkins = 0; }

  if(numberEkins > 0) {

    edm::LogInfo("SimG4CoreApplication") << "SteppingAction::Kill following "
					 << numberPart 
					 << " particles in " << numberEkins
					 << " volumes";
    for (unsigned int i=0; i<numberPart; ++i) {
      edm::LogInfo("SimG4CoreApplication") << "SteppingAction::Particle " << i
					   << " " << ekinParticles[i]
					   << "  Threshold = " << ekinMins[i]
					   << " MeV";
      ekinMins[i] *= CLHEP::MeV;
    }
    for (unsigned int i=0; i<numberEkins; ++i) {
      edm::LogInfo("SimG4CoreApplication") << "SteppingAction::LogVolume[" << i
					   << "] = " << ekinNames[i];
    }
  }
}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step * aStep) 
{
  if (!initialized) { initialized = initPointer(); }
  m_g4StepSignal(aStep);

  G4Track * theTrack = aStep->GetTrack();
  bool ok = (theTrack->GetTrackStatus() == fAlive);
  G4StepPoint* postStep = aStep->GetPostStepPoint();
  if(ok && postStep->GetPhysicalVolume() != 0) {

    G4StepPoint* preStep = aStep->GetPreStepPoint();
    const G4Region* theRegion = 
      preStep->GetPhysicalVolume()->GetLogicalVolume()->GetRegion();

    // kill in dead regions
    if(ok && 0 < ndeadRegions) { ok = killInsideDeadRegion(theTrack, theRegion); }

    // kill out of time
    if(ok) { ok = catchLongLived(theTrack, theRegion); }

    // kill low-energy in volumes on demand
    if(ok && numberEkins > 0) { ok = killLowEnergy(aStep); }

    // kill low-energy in vacuum
    G4double kinEnergy = theTrack->GetKineticEnergy();
    if(ok && killBeamPipe && kinEnergy < theCriticalEnergyForVacuum
	&& theTrack->GetDefinition()->GetPDGCharge() != 0.0 && kinEnergy > 0.0
        && theTrack->GetNextVolume()->GetLogicalVolume()->GetMaterial()->GetDensity() 
       <= theCriticalDensity) {
      theTrack->SetTrackStatus(fStopAndKill);
#ifdef DebugLog
      PrintKilledTrack(theTrack, "LE in vacuum"); 
#endif
      ok = false;
    }

    // check transition tracker/calo
    if(ok) {

      if(isThisVolume(preStep->GetTouchable(),tracker) &&
	 isThisVolume(postStep->GetTouchable(),calo)) {

	math::XYZVectorD pos((preStep->GetPosition()).x(),
			     (preStep->GetPosition()).y(),
			     (preStep->GetPosition()).z());
      
	math::XYZTLorentzVectorD mom((preStep->GetMomentum()).x(),
				     (preStep->GetMomentum()).y(),
				     (preStep->GetMomentum()).z(),
				     preStep->GetTotalEnergy());
      
	uint32_t id = theTrack->GetTrackID();
      
	std::pair<math::XYZVectorD,math::XYZTLorentzVectorD> p(pos,mom);
	eventAction_->addTkCaloStateInfo(id,p);
      }
    }
  }
}

bool SteppingAction::killInsideDeadRegion(G4Track * theTrack, 
					  const G4Region* reg) const
{
  bool alive = true;
  for(unsigned int i=0; i<ndeadRegions; ++i) {
    if(reg == deadRegions[i]) {
      alive = false;    
      theTrack->SetTrackStatus(fStopAndKill);
#ifdef DebugLog
      PrintKilledTrack(theTrack, "dead region"); 
#endif
      break;
    }
  }
  return alive;
}

bool SteppingAction::catchLongLived(G4Track* theTrack, const G4Region* reg) const
{
  bool flag   = true;
  double tofM = maxTrackTime;

  if(numberTimes > 0) {
    for (unsigned int i=0; i<numberTimes; ++i) {
      if (reg == maxTimeRegions[i]) {
	tofM = maxTrackTimes[i];
	break;
      }
    }
  }
  if (theTrack->GetGlobalTime() > tofM) {
    theTrack->SetTrackStatus(fStopAndKill);
#ifdef DebugLog
    PrintKilledTrack(theTrack, "out of time"); 
#endif
    flag = false;
  }
  return flag;
}

bool SteppingAction::killLowEnergy(const G4Step * aStep) const
{
  bool ok = true;
  bool flag = false;
  G4LogicalVolume* lv = 
    aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  for (unsigned int i=0; i<numberEkins; ++i) {
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
    for (unsigned int i=0; i<numberPart; ++i) {
      if (pCode == ekinPDG[i]) {
	ekinM = ekinMins[i];
	break;
      }
    }
    if (ekin <= ekinM) {
      track->SetTrackStatus(fStopAndKill);
#ifdef DebugLog
      PrintKilledTrack(track, "low-energy");
#endif
      ok = false;
    }
  }
  return ok;
}

bool SteppingAction::isThisVolume(const G4VTouchable* touch, 
				  G4VPhysicalVolume* pv) const
{
  bool res = false;
  int level = (touch->GetHistoryDepth())+1;
  if (level >= 3) { res = (touch->GetVolume(level - 3) == pv); }
  return res;
}

bool SteppingAction::initPointer() 
{
  const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
  if (pvs) {
    std::vector<G4VPhysicalVolume*>::const_iterator pvcite;
    for (pvcite = pvs->begin(); pvcite != pvs->end(); ++pvcite) {
      if ((*pvcite)->GetName() == "Tracker") tracker = (*pvcite);
      if ((*pvcite)->GetName() == "CALO")    calo    = (*pvcite);
      if (tracker && calo) break;
    }
    if (tracker || calo) {
      edm::LogInfo("SimG4CoreApplication") << "Pointer for Tracker " << tracker
					   << " and for Calo " << calo;
      if (tracker) LogDebug("SimG4CoreApplication") << "Tracker vol name "
						    << tracker->GetName();
      if (calo)    LogDebug("SimG4CoreApplication") << "Calorimeter vol name "
						    << calo->GetName();
    }
  }

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  if (numberEkins > 0) {
    if (lvs) {
      ekinVolumes.resize(numberEkins, 0);
      std::vector<G4LogicalVolume*>::const_iterator lvcite;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); ++lvcite) {
	for (unsigned int i=0; i<numberEkins; ++i) {
	  if ((*lvcite)->GetName() == (G4String)(ekinNames[i])) {
	    ekinVolumes[i] = (*lvcite);
	    break;
	  }
	}
      }
    }
    for (unsigned int i=0; i<numberEkins; ++i) {
      edm::LogInfo("SimG4CoreApplication") << ekinVolumes[i]->GetName()
					   <<" with pointer " << ekinVolumes[i];
    }
  }

  if(numberPart > 0) {
    G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
    G4String partName;
    ekinPDG.resize(numberPart, 0);
    for (unsigned int i=0; i<numberPart; ++i) {
      ekinPDG[i] = 
	theParticleTable->FindParticle(partName=ekinParticles[i])->GetPDGEncoding();
      edm::LogInfo("SimG4CoreApplication") << "Particle " << ekinParticles[i]
					   << " with PDG code " << ekinPDG[i]
					   << " and KE cut off " 
					   << ekinMins[i]/MeV << " MeV";
    }
  }

  const G4RegionStore * rs = G4RegionStore::GetInstance();
  if (numberTimes > 0) {
    maxTimeRegions.resize(numberTimes, 0);
    std::vector<G4Region*>::const_iterator rcite;
    for (rcite = rs->begin(); rcite != rs->end(); ++rcite) {
      for (unsigned int i=0; i<numberTimes; ++i) {
	if ((*rcite)->GetName() == (G4String)(maxTimeNames[i])) {
	  maxTimeRegions[i] = (*rcite);
	  break;
	}
      }
    }
  }
  if (ndeadRegions > 0) {
    deadRegions.resize(ndeadRegions, 0);
    std::vector<G4Region*>::const_iterator rcite;
    for (rcite = rs->begin(); rcite != rs->end(); ++rcite) {
      for (unsigned int i=0; i<ndeadRegions; ++i) {
	if ((*rcite)->GetName() == (G4String)(deadRegionNames[i])) {
	  deadRegions[i] = (*rcite);
	  break;
	}
      }
    }
  }
  return true;
}

void SteppingAction::PrintKilledTrack(const G4Track* aTrack, 
				      const std::string& typ) const
{
  std::string vname = "";
  std::string rname = "";
  G4VPhysicalVolume* pv = aTrack->GetNextVolume();
  if(pv) { 
    vname = pv->GetLogicalVolume()->GetName(); 
    rname = pv->GetLogicalVolume()->GetRegion()->GetName(); 
  }

  edm::LogInfo("SimG4CoreApplication") 
    << "Track #" << aTrack->GetTrackID()
    << " " << aTrack->GetDefinition()->GetParticleName()
    << " E(MeV)= " << aTrack->GetKineticEnergy()/MeV 
    << " is killed due to " << typ
    << " inside LV: " << vname << " (" << rname
    << ") at " << aTrack->GetPosition();
}
