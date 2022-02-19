#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingCalo.h"

#include "G4LogicalVolumeStore.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include <iostream>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class HGCalTBMBProducer : public SimProducer,
                          public Observer<const BeginOfTrack*>,
                          public Observer<const G4Step*>,
                          public Observer<const EndOfTrack*> {
public:
  HGCalTBMBProducer(const edm::ParameterSet&);
  HGCalTBMBProducer(const HGCalTBMBProducer&) = delete;                   // stop default
  const HGCalTBMBProducer& operator=(const HGCalTBMBProducer&) = delete;  // ...
  ~HGCalTBMBProducer() override = default;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void update(const BeginOfTrack*) override;
  void update(const G4Step*) override;
  void update(const EndOfTrack*) override;

  bool stopAfter(const G4Step*);
  int findVolume(const G4VTouchable* touch, bool stop) const;

  const edm::ParameterSet m_p;
  const std::vector<std::string> listNames_;
  const std::string stopName_;
  const double stopZ_;
  const unsigned int nList_;
  MaterialAccountingCaloCollection matcoll_;
  std::vector<double> radLen_, intLen_, stepLen_;
};

HGCalTBMBProducer::HGCalTBMBProducer(const edm::ParameterSet& p)
    : m_p(p.getParameter<edm::ParameterSet>("HGCalTBMB")),
      listNames_(m_p.getParameter<std::vector<std::string> >("DetectorNames")),
      stopName_(m_p.getParameter<std::string>("StopName")),
      stopZ_(m_p.getParameter<double>("MaximumZ")),
      nList_(listNames_.size()) {
  edm::LogVerbatim("HGCSim") << "HGCalTBMBProducer initialized for " << nList_ << " volumes";
  for (unsigned int k = 0; k < nList_; ++k)
    edm::LogVerbatim("HGCSim") << " [" << k << "] " << listNames_[k];
  edm::LogVerbatim("HGCSim") << "Stop after " << stopZ_ << " or reaching volume " << stopName_;

  produces<MaterialAccountingCaloCollection>("HGCalTBMB");
}

void HGCalTBMBProducer::produce(edm::Event& e, const edm::EventSetup&) {
  std::unique_ptr<MaterialAccountingCaloCollection> hgc(new MaterialAccountingCaloCollection);
  for (auto const& mbc : matcoll_) {
    hgc->emplace_back(mbc);
  }
  e.put(std::move(hgc), "HGCalTBMB");
}

void HGCalTBMBProducer::update(const BeginOfTrack* trk) {
  radLen_ = std::vector<double>(nList_ + 1, 0);
  intLen_ = std::vector<double>(nList_ + 1, 0);
  stepLen_ = std::vector<double>(nList_ + 1, 0);

#ifdef EDM_ML_DEBUG
  const G4Track* aTrack = (*trk)();  // recover G4 pointer if wanted
  const G4ThreeVector& mom = aTrack->GetMomentum();
  double theEnergy = aTrack->GetTotalEnergy();
  int theID = (int)(aTrack->GetDefinition()->GetPDGEncoding());
  edm::LogVerbatim("HGCSim") << "HGCalTBMBProducer: Track " << aTrack->GetTrackID() << " Code " << theID << " Energy "
                             << theEnergy / CLHEP::GeV << " GeV; Momentum " << mom;
#endif
}

void HGCalTBMBProducer::update(const G4Step* aStep) {
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
  edm::LogVerbatim("HGCSim") << "HGCalTBMBProducer::Step in " << touch->GetVolume(0)->GetLogicalVolume()->GetName()
                             << " Index " << indx << " Step " << step << " RadL " << step / radl << " IntL "
                             << step / intl;
#endif

  if (stopAfter(aStep)) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus(fStopAndKill);
  }
}

void HGCalTBMBProducer::update(const EndOfTrack* trk) {
  MaterialAccountingCalo matCalo;
  matCalo.m_eta = 0;
  matCalo.m_phi = 0;
  for (unsigned int ii = 0; ii <= nList_; ++ii) {
    matCalo.m_stepLen.emplace_back(stepLen_[ii]);
    matCalo.m_radLen.emplace_back(radLen_[ii]);
    matCalo.m_intLen.emplace_back(intLen_[ii]);
#ifdef EDM_ML_DEBUG
    std::string name("Total");
    if (ii < nList_)
      name = listNames_[ii];
    edm::LogVerbatim("HGCSim") << "HGCalTBMBProducer::Volume[" << ii << "]: " << name << " == Step " << stepLen_[ii]
                               << " RadL " << radLen_[ii] << " IntL " << intLen_[ii];
#endif
  }
  matcoll_.emplace_back(matCalo);
}

bool HGCalTBMBProducer::stopAfter(const G4Step* aStep) {
  bool flag(false);
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
  if (aStep->GetPostStepPoint() != nullptr)
    hitPoint = aStep->GetPostStepPoint()->GetPosition();
  double zz = hitPoint.z();

  if ((findVolume(touch, true) == 0) || (zz > stopZ_))
    flag = true;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << " HGCalTBMBProducer::Name " << touch->GetVolume(0)->GetName() << " z " << zz << " Flag"
                             << flag;
#endif
  return flag;
}

int HGCalTBMBProducer::findVolume(const G4VTouchable* touch, bool stop) const {
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

DEFINE_SIMWATCHER(HGCalTBMBProducer);
