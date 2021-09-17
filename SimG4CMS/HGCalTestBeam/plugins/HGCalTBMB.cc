#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include <TH1F.h>

#include <iostream>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class HGCalTBMB : public SimWatcher,
                  public Observer<const BeginOfTrack*>,
                  public Observer<const G4Step*>,
                  public Observer<const EndOfTrack*> {
public:
  HGCalTBMB(const edm::ParameterSet&);
  HGCalTBMB(const HGCalTBMB&) = delete;                   // stop default
  const HGCalTBMB& operator=(const HGCalTBMB&) = delete;  // ...
  ~HGCalTBMB() override;

private:
  void update(const BeginOfTrack*) override;
  void update(const G4Step*) override;
  void update(const EndOfTrack*) override;

  bool stopAfter(const G4Step*);
  int findVolume(const G4VTouchable* touch, bool stop) const;

  std::vector<std::string> listNames_;
  std::string stopName_;
  double stopZ_;
  unsigned int nList_;
  std::vector<double> radLen_, intLen_, stepLen_;
  std::vector<TH1D*> me100_, me200_, me300_;
};

HGCalTBMB::HGCalTBMB(const edm::ParameterSet& p) {
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("HGCalTBMB");
  listNames_ = m_p.getParameter<std::vector<std::string> >("DetectorNames");
  stopName_ = m_p.getParameter<std::string>("StopName");
  stopZ_ = m_p.getParameter<double>("MaximumZ");
  nList_ = listNames_.size();
  edm::LogVerbatim("HGCSim") << "HGCalTBMB initialized for " << nList_ << " volumes";
  for (unsigned int k = 0; k < nList_; ++k)
    edm::LogVerbatim("HGCSim") << " [" << k << "] " << listNames_[k];
  edm::LogVerbatim("HGCSim") << "Stop after " << stopZ_ << " or reaching volume " << stopName_;

  edm::Service<TFileService> tfile;
  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char name[20], title[80];
  TH1D* hist;
  for (unsigned int i = 0; i <= nList_; i++) {
    std::string named = (i == nList_) ? "Total" : listNames_[i];
    sprintf(name, "RadL%d", i);
    sprintf(title, "MB(X0) for (%s)", named.c_str());
    hist = tfile->make<TH1D>(name, title, 100000, 0.0, 100.0);
    hist->Sumw2(true);
    me100_.push_back(hist);
    sprintf(name, "IntL%d", i);
    sprintf(title, "MB(L0) for (%s)", named.c_str());
    hist = tfile->make<TH1D>(name, title, 100000, 0.0, 10.0);
    hist->Sumw2(true);
    me200_.push_back(hist);
    sprintf(name, "StepL%d", i);
    sprintf(title, "MB(Step) for (%s)", named.c_str());
    hist = tfile->make<TH1D>(name, title, 100000, 0.0, 50000.0);
    hist->Sumw2(true);
    me300_.push_back(hist);
  }
  edm::LogVerbatim("HGCSim") << "HGCalTBMB: Booking user histos done ===";
}

HGCalTBMB::~HGCalTBMB() {}

void HGCalTBMB::update(const BeginOfTrack* trk) {
  radLen_ = std::vector<double>(nList_ + 1, 0);
  intLen_ = std::vector<double>(nList_ + 1, 0);
  stepLen_ = std::vector<double>(nList_ + 1, 0);

#ifdef EDM_ML_DEBUG
  const G4Track* aTrack = (*trk)();  // recover G4 pointer if wanted
  const G4ThreeVector& mom = aTrack->GetMomentum();
  double theEnergy = aTrack->GetTotalEnergy();
  int theID = (int)(aTrack->GetDefinition()->GetPDGEncoding());
  edm::LogVerbatim("HGCSim") << "MaterialBudgetHcalHistos: Track " << aTrack->GetTrackID() << " Code " << theID
                             << " Energy " << theEnergy / CLHEP::GeV << " GeV; Momentum " << mom;
#endif
}

void HGCalTBMB::update(const G4Step* aStep) {
  G4Material* material = aStep->GetPreStepPoint()->GetMaterial();
  double step = aStep->GetStepLength();
  double radl = material->GetRadlen();
  double intl = material->GetNuclearInterLength();

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int indx = findVolume(touch, false);

  if (indx >= 0) {
    stepLen_[indx] += step;
    radLen_[indx] += (step / radl);
    intLen_[indx] += (step / intl);
  }
  stepLen_[nList_] += step;
  radLen_[nList_] += (step / radl);
  intLen_[nList_] += (step / intl);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTBMB::Step in " << touch->GetVolume(0)->GetLogicalVolume()->GetName() << " Index "
                             << indx << " Step " << step << " RadL " << step / radl << " IntL " << step / intl;
#endif

  if (stopAfter(aStep)) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus(fStopAndKill);
  }
}

void HGCalTBMB::update(const EndOfTrack* trk) {
  for (unsigned int ii = 0; ii <= nList_; ++ii) {
    me100_[ii]->Fill(radLen_[ii]);
    me200_[ii]->Fill(intLen_[ii]);
    me300_[ii]->Fill(stepLen_[ii]);
#ifdef EDM_ML_DEBUG
    std::string name("Total");
    if (ii < nList_)
      name = listNames_[ii];
    edm::LogVerbatim("HGCSim") << "HGCalTBMB::Volume[" << ii << "]: " << name << " == Step " << stepLen_[ii] << " RadL "
                               << radLen_[ii] << " IntL " << intLen_[ii];
#endif
  }
}

bool HGCalTBMB::stopAfter(const G4Step* aStep) {
  bool flag(false);
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
  if (aStep->GetPostStepPoint() != nullptr)
    hitPoint = aStep->GetPostStepPoint()->GetPosition();
  double zz = hitPoint.z();

  if ((findVolume(touch, true) == 0) || (zz > stopZ_))
    flag = true;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << " HGCalTBMB::Name " << touch->GetVolume(0)->GetName() << " z " << zz << " Flag" << flag;
#endif
  return flag;
}

int HGCalTBMB::findVolume(const G4VTouchable* touch, bool stop) const {
  int ivol = -1;
  int level = (touch->GetHistoryDepth()) + 1;
  for (int ii = 0; ii < level; ii++) {
    std::string name = touch->GetVolume(ii)->GetName();
    if (stop) {
      if (strcmp(name.c_str(), stopName_.c_str()) == 0)
        ivol = 0;
    } else {
      for (unsigned int k = 0; k < nList_; ++k) {
        if (strcmp(name.c_str(), listNames_[k].c_str()) == 0) {
          ivol = k;
          break;
        }
      }
    }
    if (ivol >= 0)
      break;
  }
  return ivol;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

DEFINE_SIMWATCHER(HGCalTBMB);
