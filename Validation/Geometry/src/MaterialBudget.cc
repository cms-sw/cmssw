#include "Validation/Geometry/interface/MaterialBudget.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include <iostream>
//#define DebugLog

MaterialBudget::MaterialBudget(const edm::ParameterSet& p) {
  
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MaterialBudget");
  detTypes     = m_p.getParameter<std::vector<std::string> >("DetectorTypes");
  constituents = m_p.getParameter<std::vector<int> >("Constituents");
  stackOrder   = m_p.getParameter<std::vector<int> >("StackOrder");
  detNames     = m_p.getParameter<std::vector<std::string> >("DetectorNames");
  detLevels    = m_p.getParameter<std::vector<int> >("DetectorLevels");
  etaRegions   = m_p.getParameter<std::vector<double> >("EtaBoundaries");
  regionTypes  = m_p.getParameter<std::vector<int> >("RegionTypes");
  boundaries   = m_p.getParameter<std::vector<double> >("Boundaries");
  edm::LogInfo("MaterialBudget") << "MaterialBudget initialized for "
				 << detTypes.size() << " detector types";
  unsigned int nc = 0;
  for (unsigned int ii=0; ii<detTypes.size(); ++ii) {
    edm::LogInfo("MaterialBudget") << "Type[" << ii << "] : " << detTypes[ii]
				   << " with " << constituents[ii] <<", order "
				   << stackOrder[ii] << " constituents --> ";
    for (int kk=0; kk<constituents[ii]; ++kk) {
      std::string name = "Unknown"; int level = -1;
      if (nc < detNames.size()) {
	name = detNames[nc]; level = detLevels[nc]; ++nc;
      }
      edm::LogInfo("MaterialBudget") << "    Constituent[" << kk << "]: "
				     << name << " at level " << level;
    }
  }
  edm::LogInfo("MaterialBudget") << "MaterialBudget Stop condition for "
				 << etaRegions.size() << " eta regions";
  for (unsigned int ii=0; ii<etaRegions.size(); ++ii) {
    edm::LogInfo("MaterialBudget") << "Region[" << ii << "] : Eta < " 
				   << etaRegions[ii] << " boundary type "
				   << regionTypes[ii] << " limit "
				   << boundaries[ii];
  }
  book(m_p);
}

MaterialBudget::~MaterialBudget() {
}

void MaterialBudget::update(const BeginOfRun* ) {

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;

  unsigned int kount=detNames.size();
  for (unsigned int ii=0; ii<kount; ++ii) 
    logVolumes.push_back(0);

  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
    for (unsigned int ii=0; ii<detNames.size(); ++ii) {
      if (strcmp(detNames[ii].c_str(),(*lvcite)->GetName().c_str()) == 0) {
	logVolumes[ii] = (*lvcite);
	kount--;
	break;
      }
    }
    if (kount <= 0) break;
  }
  edm::LogInfo("MaterialBudget") << "MaterialBudget::Finds " 
				 << detNames.size()-kount << " out of "
				 << detNames.size() << " LV addresses";
  for (unsigned int ii=0; ii<detNames.size(); ++ii) {
    std::string name("Unknown");
    if (logVolumes[ii] != 0)  name = logVolumes[ii]->GetName();
    edm::LogInfo("MaterialBudget") << "LV[" << ii << "] : " << detNames[ii]
				   << " Address " << logVolumes[ii] << " | " 
				   << name;
  }

  for (unsigned int ii=0; ii<(detTypes.size()+1); ++ii) {
    stepLen.push_back(0); radLen.push_back(0); intLen.push_back(0);
  }
  stackOrder.push_back(0);
}

void MaterialBudget::update(const BeginOfTrack* trk) {

  for (unsigned int ii=0; ii<(detTypes.size()+1); ++ii) {
    stepLen[ii] = 0; radLen[ii] = 0; intLen[ii] = 0;
  }

  const G4Track * aTrack = (*trk)(); // recover G4 pointer if wanted
  const G4ThreeVector& mom = aTrack->GetMomentum() ;
  if (mom.theta() != 0 ) {
    eta = mom.eta();
  } else {
    eta = -99;
  }
  phi = mom.phi();
  stepT = 0;

#ifdef DebugLog
  double theEnergy = aTrack->GetTotalEnergy();
  int    theID     = (int)(aTrack->GetDefinition()->GetPDGEncoding());
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Track "
				 << aTrack->GetTrackID() << " Code " << theID
				 << " Energy " <<theEnergy/CLHEP::GeV
				 << " GeV; Eta " << eta << " Phi " 
				 << phi/CLHEP::deg << " PT "
				 << mom.perp()/CLHEP::GeV << " GeV *****";
#endif
}
 
void MaterialBudget::update(const G4Step* aStep) {

  //---------- each step
  G4Material * material = aStep->GetPreStepPoint()->GetMaterial();
  double step    = aStep->GetStepLength();
  double radl    = material->GetRadlen();
  double intl    = material->GetNuclearInterLength();

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  unsigned int indx = detTypes.size();
  unsigned int nc = 0;
  for (unsigned int ii=0; ii<detTypes.size(); ++ii) {
    for (int kk=0; kk<constituents[ii]; ++kk) {
      if (detLevels[nc+kk] <= (int)((touch->GetHistoryDepth())+1)) {
	int jj = (int)((touch->GetHistoryDepth())+1)-detLevels[nc+kk];
        if ((touch->GetVolume(jj)->GetLogicalVolume()) == logVolumes[nc+kk]) {
	  indx = ii;
	  break;
	}
      }
    }
    nc += (unsigned int)(constituents[ii]);
    if (indx == ii) break;
  }

  stepT         += step;
  stepLen[indx]  = stepT;
  radLen[indx]  += (step/radl);
  intLen[indx]  += (step/intl);
#ifdef DebugLog
  edm::LogInfo("MaterialBudget") << "MaterialBudget::Step in "
				 << touch->GetVolume(0)->GetLogicalVolume()->GetName()
				 << " Index " << indx <<" Step " << step 
				 << " RadL " << step/radl << " IntL " 
				 << step/intl;
#endif
  //----- Stop tracking after selected position
  if (stopAfter(aStep)) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus( fStopAndKill );
  }
}


void MaterialBudget::update(const EndOfTrack* trk) {

  for (unsigned int ii=0; ii<detTypes.size(); ++ii) {
    for (unsigned int jj=0; jj<=detTypes.size(); ++jj) {
      if (stackOrder[jj] == (int)(ii+1)) {
	for (unsigned int kk=0; kk<=detTypes.size(); ++kk) {
	  if (stackOrder[kk] == (int)(ii)) {
	    radLen[jj]  += radLen[kk];
	    intLen[jj]  += intLen[kk];
#ifdef DebugLog
	    edm::LogInfo("MaterialBudget") << "MaterialBudget::Add " 
					   << kk << ":" << stackOrder[kk] 
					   << " to " << jj <<":" 
					   << stackOrder[jj] <<" RadL "
					   << radLen[kk] << " : " << radLen[jj]
					   << " IntL " << intLen[kk] << " : "
					   << intLen[jj] <<" StepL " 
					   << stepLen[kk]<< " : " 
					   << stepLen[jj];
#endif
	    break;
	  }
	}
	break;
      }
    }
  }

  for (unsigned int ii=0; ii<=detTypes.size(); ++ii) {
    me100[ii]->Fill(eta, radLen[ii]);
    me200[ii]->Fill(eta, intLen[ii]);
    me300[ii]->Fill(eta, stepLen[ii]);
    me400[ii]->Fill(phi, radLen[ii]);
    me500[ii]->Fill(phi, intLen[ii]);
    me600[ii]->Fill(phi, stepLen[ii]);
#ifdef DebugLog
    std::string name("Unknown");
    if (ii < detTypes.size()) name = detTypes[ii];
    edm::LogInfo("MaterialBudget") << "MaterialBudget::Volume[" << ii
				   << "]: " << name << " eta " << eta 
				   << " == Step " << stepLen[ii] << " RadL " 
				   << radLen[ii] << " IntL " << intLen[ii];
#endif
  }
}

void MaterialBudget::book(const edm::ParameterSet& m_p) {

  // Book histograms
  edm::Service<TFileService> tfile;

  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  int    binEta  = m_p.getUntrackedParameter<int>("NBinEta", 320);
  int    binPhi  = m_p.getUntrackedParameter<int>("NBinPhi", 180);
  double minEta  = m_p.getUntrackedParameter<double>("MinEta",-8.0);
  double maxEta  = m_p.getUntrackedParameter<double>("MaxEta", 8.0);
  double maxPhi  = CLHEP::pi;
  edm::LogInfo("MaterialBudget") << "MaterialBudget: Booking user "
                                 << "histos === with " << binEta << " bins "
                                 << "in eta from " << minEta << " to "
                                 << maxEta << " and " << binPhi << " bins "
                                 << "in phi from " << -maxPhi << " to "
                                 << maxPhi;

  char  name[20], title[80];
  std::string named;
  for (unsigned int i=0; i<=detTypes.size(); i++) {
    if (i >= detTypes.size()) named = "Unknown";
    else                      named = detTypes[i];
    sprintf(name, "%d", i+100);
    sprintf(title, "MB(X0) prof Eta in %s", named.c_str());
    me100.push_back(tfile->make<TProfile>(name,title, binEta, minEta, maxEta));
    sprintf(name, "%d", i+200);
    sprintf(title, "MB(L0) prof Eta in %s", named.c_str());
    me200.push_back(tfile->make<TProfile>(name,title, binEta, minEta, maxEta));
    sprintf(name, "%d", i+300);
    sprintf(title, "MB(Step) prof Eta in %s", named.c_str());
    me300.push_back(tfile->make<TProfile>(name,title, binEta, minEta, maxEta));
    sprintf(name, "%d", i+400);
    sprintf(title, "MB(X0) prof Phi in %s", named.c_str());
    me400.push_back(tfile->make<TProfile>(name,title, binPhi,-maxPhi, maxPhi));
    sprintf(name, "%d", i+500);
    sprintf(title, "MB(L0) prof Phi in %s", named.c_str());
    me500.push_back(tfile->make<TProfile>(name,title, binPhi,-maxPhi, maxPhi));
    sprintf(name, "%d", i+600);
    sprintf(title, "MB(Step) prof Phi in %s", named.c_str());
    me600.push_back(tfile->make<TProfile>(name,title, binPhi,-maxPhi, maxPhi));
  }

  edm::LogInfo("MaterialBudget") << "MaterialBudget: Booking user "
                                 << "histos done ===";

}

bool MaterialBudget::stopAfter(const G4Step* aStep) {

  G4ThreeVector hitPoint    = aStep->GetPreStepPoint()->GetPosition();
  if (aStep->GetPostStepPoint() != 0) 
    hitPoint = aStep->GetPostStepPoint()->GetPosition();
  double rr    = hitPoint.perp();
  double zz    = std::abs(hitPoint.z());

  bool   flag(false);
  for (unsigned int ii=0; ii<etaRegions.size(); ++ii) {
#ifdef DebugLog
    edm::LogInfo("MaterialBudget") << " MaterialBudget::Eta " << eta 
				   << " in Region[" << ii << "] with " 
				   << etaRegions[ii] << " type "   
				   << regionTypes[ii] << "|" << boundaries[ii];
#endif
    if (fabs(eta) < etaRegions[ii]) {
      if (regionTypes[ii] == 0) {
	if (rr >= boundaries[ii]-0.001) flag = true;
      } else {
	if (zz >= boundaries[ii]-0.001) flag = true;
      }
#ifdef DebugLog
      if (flag)
	edm::LogInfo("MaterialBudget") <<" MaterialBudget::Stop after R = " 
				       << rr << " and Z = " << zz;
#endif
      break;
    }
  }
#ifdef DebugLog
  edm::LogInfo("MaterialBudget") <<" MaterialBudget:: R = " << rr 
				 << " and Z = "  << zz << " Flag " << flag;
#endif
  return flag;
}
