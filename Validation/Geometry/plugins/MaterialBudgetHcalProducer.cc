#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingCalo.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include <CLHEP/Vector/LorentzVector.h>

#include "DD4hep/Filter.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace geant_units::operators;

class MaterialBudgetHcalProducer : public SimProducer,
                                   public Observer<const BeginOfEvent*>,
                                   public Observer<const BeginOfTrack*>,
                                   public Observer<const G4Step*>,
                                   public Observer<const EndOfTrack*> {
public:
  MaterialBudgetHcalProducer(const edm::ParameterSet&);
  ~MaterialBudgetHcalProducer() override = default;

  MaterialBudgetHcalProducer(const MaterialBudgetHcalProducer&) = delete;                   // stop default
  const MaterialBudgetHcalProducer& operator=(const MaterialBudgetHcalProducer&) = delete;  // stop default

  void registerConsumes(edm::ConsumesCollector) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::EventSetup const&) override;

private:
  void update(const BeginOfEvent*) override;
  void update(const BeginOfTrack*) override;
  void update(const G4Step*) override;
  void update(const EndOfTrack*) override;

  bool stopAfter(const G4Step*);
  std::vector<std::string> getNames(DDFilteredView& fv);
  std::vector<std::string> getNames(cms::DDFilteredView& fv);
  std::vector<double> getDDDArray(const std::string& str, const DDsvalues_type& sv);
  bool isSensitive(const std::string&);
  bool isItHF(const G4VTouchable*);
  bool isItEC(const std::string&);

  static const int maxSet_ = 25, maxSet2_ = 9;
  double rMax_, zMax_, etaMinP_, etaMaxP_;
  bool printSum_, fromdd4hep_;
  std::string name_;

  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvTokenDDD_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokenDD4hep_;

  std::vector<std::string> sensitives_, hfNames_, sensitiveEC_;
  std::vector<int> hfLevels_;
  MaterialAccountingCaloCollection matcoll_;
  double stepLens_[maxSet_], radLens_[maxSet_], intLens_[maxSet_];
  std::vector<std::string> matList_;
  std::vector<double> stepLength_, radLength_, intLength_;
  int id_, layer_, steps_;
  double radLen_, intLen_, stepLen_;
  double eta_, phi_;
  int nlayHB_, nlayHE_, nlayHO_, nlayHF_;
};

MaterialBudgetHcalProducer::MaterialBudgetHcalProducer(const edm::ParameterSet& p) {
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MaterialBudgetHcalProducer");
  rMax_ = m_p.getUntrackedParameter<double>("RMax", 4.5) * CLHEP::m;
  zMax_ = m_p.getUntrackedParameter<double>("ZMax", 13.0) * CLHEP::m;
  etaMinP_ = m_p.getUntrackedParameter<double>("EtaMinP", 5.2);
  etaMaxP_ = m_p.getUntrackedParameter<double>("EtaMaxP", 0.0);
  fromdd4hep_ = m_p.getUntrackedParameter<bool>("Fromdd4hep", false);
  printSum_ = m_p.getUntrackedParameter<bool>("PrintSummary", false);
  name_ = m_p.getUntrackedParameter<std::string>("Name", "Hcal");
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalProducer initialized with rMax " << rMax_ << " mm and zMax "
                                     << zMax_ << " mm printSummary is set to " << printSum_ << " and Fromdd4hep to "
                                     << fromdd4hep_;

  produces<MaterialAccountingCaloCollection>(Form("%sMatBCalo", name_.c_str()));
}

void MaterialBudgetHcalProducer::registerConsumes(edm::ConsumesCollector cc) {
  if (fromdd4hep_)
    cpvTokenDD4hep_ = cc.esConsumes<edm::Transition::BeginRun>();
  else
    cpvTokenDDD_ = cc.esConsumes<edm::Transition::BeginRun>();
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalProducer: Initialize the token for CompactView";
}

void MaterialBudgetHcalProducer::produce(edm::Event& e, const edm::EventSetup&) {
  std::unique_ptr<MaterialAccountingCaloCollection> hgc(new MaterialAccountingCaloCollection);
  for (auto const& mbc : matcoll_) {
    hgc->emplace_back(mbc);
  }
  e.put(std::move(hgc), Form("%sMatBCalo", name_.c_str()));
}

void MaterialBudgetHcalProducer::beginRun(edm::EventSetup const& es) {
  //----- Check that selected volumes are indeed part of the geometry
  // Numbering From DDD
  if (fromdd4hep_) {
    const cms::DDCompactView cpv = es.getData(cpvTokenDD4hep_);
    constexpr int32_t addLevel = 1;
    std::string attribute = "ReadOutName";
    std::string value = "HcalHits";
    const cms::DDFilter filter1(attribute, value);
    cms::DDFilteredView fv1(cpv, filter1);
    std::vector<std::string> names = getNames(fv1);
    for (auto& name : names) {
      std::string namx = (name.find('_') == std::string::npos) ? name : name.substr(0, name.find('_'));
      if (std::find(sensitives_.begin(), sensitives_.end(), namx) == sensitives_.end())
        sensitives_.emplace_back(namx);
    }
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Names to be tested for " << attribute
                                           << " = " << value << " has " << sensitives_.size() << " elements";
    for (unsigned int i = 0; i < sensitives_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalProducer: sensitives[" << i << "] = " << sensitives_[i];
    attribute = "Volume";
    value = "HF";
    const cms::DDFilter filter2(attribute, value);
    cms::DDFilteredView fv2(cpv, filter2);
    std::vector<int> temp = fv2.get<std::vector<int> >("hf", "Levels");
    hfNames_ = getNames(fv2);
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Names to be tested for " << attribute
                                           << " = " << value << " has " << hfNames_.size() << " elements";
    for (unsigned int i = 0; i < hfNames_.size(); i++) {
      hfLevels_.push_back(temp[i] + addLevel);
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalProducer:  HF[" << i << "] = " << hfNames_[i] << " at level " << hfLevels_[i];
    }

    const std::string ecalRO[2] = {"EcalHitsEB", "EcalHitsEE"};
    attribute = "ReadOutName";
    for (int k = 0; k < 2; k++) {
      value = ecalRO[k];
      const cms::DDFilter filter(attribute, value);
      cms::DDFilteredView fv(cpv, filter);
      std::vector<std::string> senstmp = getNames(fv);
      edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Names to be tested for " << attribute
                                             << " = " << value << " has " << senstmp.size() << " elements";
      for (unsigned int i = 0; i < senstmp.size(); i++) {
        std::string name = senstmp[i].substr(0, 4);
        if (std::find(sensitiveEC_.begin(), sensitiveEC_.end(), name) == sensitiveEC_.end())
          sensitiveEC_.push_back(name);
      }
    }
    for (unsigned int i = 0; i < sensitiveEC_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalProducer:sensitiveEC[" << i << "] = " << sensitiveEC_[i];
  } else {  // if not from dd4hep --> ddd
    const DDCompactView& cpv = es.getData(cpvTokenDDD_);
    constexpr int32_t addLevel = 0;
    std::string attribute = "ReadOutName";
    std::string value = "HcalHits";
    DDSpecificsMatchesValueFilter filter1{DDValue(attribute, value, 0)};
    DDFilteredView fv1(cpv, filter1);
    std::vector<std::string> names = getNames(fv1);
    for (auto& name : names) {
      std::string namx = (name.find('_') == std::string::npos) ? name : name.substr(0, name.find('_'));
      if (std::find(sensitives_.begin(), sensitives_.end(), namx) == sensitives_.end())
        sensitives_.emplace_back(namx);
    }
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Names to be tested for " << attribute
                                           << " = " << value << " has " << sensitives_.size() << " elements";
    for (unsigned int i = 0; i < sensitives_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalProducer: sensitives[" << i << "] = " << sensitives_[i];
    attribute = "Volume";
    value = "HF";
    DDSpecificsMatchesValueFilter filter2{DDValue(attribute, value, 0)};
    DDFilteredView fv2(cpv, filter2);
    hfNames_ = getNames(fv2);
    fv2.firstChild();
    DDsvalues_type sv(fv2.mergedSpecifics());
    std::vector<double> temp = getDDDArray("Levels", sv);
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Names to be tested for " << attribute
                                           << " = " << value << " has " << hfNames_.size() << " elements";
    for (unsigned int i = 0; i < hfNames_.size(); i++) {
      int level = static_cast<int>(temp[i]);
      hfLevels_.push_back(level + addLevel);
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalProducer:  HF[" << i << "] = " << hfNames_[i] << " at level " << hfLevels_[i];
    }

    const std::string ecalRO[2] = {"EcalHitsEB", "EcalHitsEE"};
    attribute = "ReadOutName";
    for (int k = 0; k < 2; k++) {
      value = ecalRO[k];
      DDSpecificsMatchesValueFilter filter3{DDValue(attribute, value, 0)};
      DDFilteredView fv3(cpv, filter3);
      std::vector<std::string> senstmp = getNames(fv3);
      edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Names to be tested for " << attribute
                                             << " = " << value << " has " << senstmp.size() << " elements";
      for (unsigned int i = 0; i < senstmp.size(); i++) {
        std::string name = senstmp[i].substr(0, 4);
        if (std::find(sensitiveEC_.begin(), sensitiveEC_.end(), name) == sensitiveEC_.end())
          sensitiveEC_.push_back(name);
      }
    }
    for (unsigned int i = 0; i < sensitiveEC_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalProducer:sensitiveEC[" << i << "] = " << sensitiveEC_[i];
  }
}

void MaterialBudgetHcalProducer::update(const BeginOfEvent*) { matcoll_.clear(); }

void MaterialBudgetHcalProducer::update(const BeginOfTrack* trk) {
  const G4Track* aTrack = (*trk)();  // recover G4 pointer if wanted

  id_ = layer_ = steps_ = 0;
  radLen_ = intLen_ = stepLen_ = 0;
  nlayHB_ = nlayHE_ = nlayHF_ = nlayHO_ = 0;
  for (int i = 0; i < maxSet_; ++i)
    stepLens_[i] = radLens_[i] = intLens_[i] = 0;

  const G4ThreeVector& dir = aTrack->GetMomentum();
  if (dir.theta() != 0) {
    eta_ = dir.eta();
  } else {
    eta_ = -99;
  }
  phi_ = dir.phi();
  edm::LogVerbatim("MaterialBudget") << "Start track with (eta, phi) = (" << eta_ << ", " << phi_ << " Energy "
                                     << aTrack->GetTotalEnergy() << " and ID "
                                     << (aTrack->GetDefinition()->GetPDGEncoding());

  if (printSum_) {
    matList_.clear();
    stepLength_.clear();
    radLength_.clear();
    intLength_.clear();
  }
}

void MaterialBudgetHcalProducer::update(const G4Step* aStep) {
  //---------- each step
  G4Material* material = aStep->GetPreStepPoint()->GetMaterial();
  double step = aStep->GetStepLength();
  double radl = material->GetRadlen();
  double intl = material->GetNuclearInterLength();
  double density = convertUnitsTo(1._g_per_cm3, material->GetDensity());

  int idOld = id_;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  std::string name = (static_cast<std::string>(dd4hep::dd::noNamespace(touch->GetVolume(0)->GetName())));
  std::string matName = (static_cast<std::string>(dd4hep::dd::noNamespace(material->GetName())));
  if (printSum_) {
    bool found = false;
    for (unsigned int ii = 0; ii < matList_.size(); ii++) {
      if (matList_[ii] == matName) {
        stepLength_[ii] += step;
        radLength_[ii] += (step / radl);
        intLength_[ii] += (step / intl);
        found = true;
        break;
      }
    }
    if (!found) {
      matList_.push_back(matName);
      stepLength_.push_back(step);
      radLength_.push_back(step / radl);
      intLength_.push_back(step / intl);
    }
    if ((std::abs(eta_) >= etaMinP_) && (std::abs(eta_) <= etaMaxP_))
      edm::LogVerbatim("MaterialBudget") << "Volume " << name << " id " << id_ << ":" << idOld << " Step " << step
                                         << " Material " << matName << " Old Length " << stepLen_ << " X0 "
                                         << step / radl << ":" << radLen_ << " Lambda " << step / intl << ":"
                                         << intLen_;
  } else {
    if ((std::abs(eta_) >= etaMinP_) && (std::abs(eta_) <= etaMaxP_))
      edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalProducer: Step at " << name << " id " << id_ << ":"
                                         << idOld << " Length " << step << " in " << matName << " of density "
                                         << density << " g/cc; Radiation Length " << radl << " mm; Interaction Length "
                                         << intl << " mm\n                          Position "
                                         << aStep->GetPreStepPoint()->GetPosition() << " Cylindrical R "
                                         << aStep->GetPreStepPoint()->GetPosition().perp() << " Length (so far) "
                                         << stepLen_ << " L/X0 " << step / radl << "/" << radLen_ << " L/Lambda "
                                         << step / intl << "/" << intLen_;
  }

  int det = 0, lay = 0;
  double abseta = std::abs(eta_);
  edm::LogVerbatim("MaterialBudgetFull") << "Volume " << name << ":" << matName << " EC:Sensitive:HF " << isItEC(name)
                                         << ":" << isSensitive(name) << ":" << isItHF(touch) << " Eta " << abseta
                                         << " HC " << ((touch->GetReplicaNumber(1)) / 1000) << ":"
                                         << ((touch->GetReplicaNumber(0) / 10) % 100 + 3) << " X0 "
                                         << (radLen_ + (step / radl)) << " Lambda " << (intLen_ + (step / intl));

  if (isItEC(name)) {
    det = 1;
    lay = 1;
  } else {
    if (isSensitive(name)) {
      if (isItHF(touch)) {
        det = 5;
        lay = 21;
        if (lay != layer_)
          ++nlayHF_;
      } else {
        det = (touch->GetReplicaNumber(1)) / 1000;
        lay = (touch->GetReplicaNumber(0) / 10) % 100 + 3;
        if (det == 4) {
          if (abseta < 1.479)
            lay = layer_ + 1;
          else
            lay--;
          if (lay < 3)
            lay = 3;
          if (lay == layer_)
            lay++;
          if (lay > 20)
            lay = 20;
          if (lay != layer_)
            ++nlayHE_;
        } else if (lay != layer_) {
          if (lay >= 20)
            ++nlayHO_;
          else
            ++nlayHB_;
        }
      }
      edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer: Det " << det << " Layer " << lay << " Eta "
                                             << eta_ << " Phi " << convertRadToDeg(phi_);
    } else if (layer_ == 1) {
      det = -1;
      lay = 2;
    }
  }
  if (det != 0) {
    if (lay != layer_) {
      id_ = lay;
      layer_ = lay;
    }
  }

  if (id_ > idOld) {
    if ((abseta >= etaMinP_) && (abseta <= etaMaxP_))
      edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalProducer: Step at " << name << " calls filHisto with "
                                         << (id_ - 1);
    stepLens_[id_ - 1] = stepLen_;
    radLens_[id_ - 1] = radLen_;
    intLens_[id_ - 1] = intLen_;
  }

  stepLen_ += step;
  radLen_ += (step / radl);
  intLen_ += (step / intl);
  if (id_ == 21) {
    if (!isItHF(aStep->GetPostStepPoint()->GetTouchable())) {
      if ((abseta >= etaMinP_) && (abseta <= etaMaxP_))
        edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalProducer: After HF in " << name << ":"
                                           << static_cast<std::string>(dd4hep::dd::noNamespace(
                                                  aStep->GetPostStepPoint()->GetTouchable()->GetVolume(0)->GetName()))
                                           << " calls fillHisto with " << id_;
      stepLens_[idOld] = stepLen_;
      radLens_[idOld] = radLen_;
      intLens_[idOld] = intLen_;
      ++id_;
      layer_ = 0;
    }
  }

  //----- Stop tracking after selected position
  if (stopAfter(aStep)) {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus(fStopAndKill);
  }
}

void MaterialBudgetHcalProducer::update(const EndOfTrack* trk) {
  MaterialAccountingCalo matCalo;
  matCalo.m_eta = eta_;
  matCalo.m_phi = phi_;
  for (int k = 0; k < maxSet_; ++k) {
    matCalo.m_stepLen.emplace_back(stepLens_[k]);
    matCalo.m_radLen.emplace_back(radLens_[k]);
    matCalo.m_intLen.emplace_back(intLens_[k]);
  }
  matCalo.m_layers.emplace_back(nlayHB_);
  matCalo.m_layers.emplace_back(nlayHE_);
  matCalo.m_layers.emplace_back(nlayHO_);
  matCalo.m_layers.emplace_back(nlayHF_);
  matcoll_.emplace_back(matCalo);
}

bool MaterialBudgetHcalProducer::stopAfter(const G4Step* aStep) {
  G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
  double rr = hitPoint.perp();
  double zz = std::abs(hitPoint.z());

  if (rr > rMax_ || zz > zMax_) {
    edm::LogVerbatim("MaterialBudget") << " MaterialBudgetHcalProducer::StopAfter R = " << rr << " and Z = " << zz;
    return true;
  } else {
    return false;
  }
}

std::vector<std::string> MaterialBudgetHcalProducer::getNames(DDFilteredView& fv) {
  std::vector<std::string> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart& log = fv.logicalPart();
    std::string namx = log.name().name();
    if (std::find(tmp.begin(), tmp.end(), namx) == tmp.end())
      tmp.push_back(namx);
    dodet = fv.next();
  }
  return tmp;
}

std::vector<std::string> MaterialBudgetHcalProducer::getNames(cms::DDFilteredView& fv) {
  std::vector<std::string> tmp;
  const std::vector<std::string> notIn = {
      "CALO", "HCal", "MBBTL", "MBBTR", "MBBTC", "MBAT", "MBBT_R1M", "MBBT_R1P", "MBBT_R1MX", "MBBT_R1PX", "VCAL"};
  while (fv.firstChild()) {
    const std::string n{fv.name().data(), fv.name().size()};
    if (std::find(notIn.begin(), notIn.end(), n) == notIn.end()) {
      std::string::size_type pos = n.find(':');
      const std::string namx = (pos == std::string::npos) ? n : std::string(n, pos + 1, n.size() - 1);
      if (std::find(tmp.begin(), tmp.end(), namx) == tmp.end())
        tmp.push_back(namx);
    }
  }
  return tmp;
}

std::vector<double> MaterialBudgetHcalProducer::getDDDArray(const std::string& str, const DDsvalues_type& sv) {
  edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalProducer:getDDDArray called for " << str;
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    edm::LogVerbatim("MaterialBudgetFull") << value;
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      throw cms::Exception("MaterialBudgetHcalProducer") << "nval = " << nval << " < 1 for array " << str << "\n";
    }

    return fvec;
  } else {
    throw cms::Exception("MaterialBudgetHcalProducer") << "cannot get array " << str << "\n";
  }
}

bool MaterialBudgetHcalProducer::isSensitive(const std::string& name) {
  std::vector<std::string>::const_iterator it = sensitives_.begin();
  std::vector<std::string>::const_iterator itEnd = sensitives_.end();
  std::string namx = (name.find('_') == std::string::npos) ? name : name.substr(0, name.find('_'));
  for (; it != itEnd; ++it)
    if (namx == *it)
      return true;
  return false;
}

bool MaterialBudgetHcalProducer::isItHF(const G4VTouchable* touch) {
  int levels = ((touch->GetHistoryDepth()) + 1);
  for (unsigned int it = 0; it < hfNames_.size(); it++) {
    if (levels >= hfLevels_[it]) {
      std::string name =
          (static_cast<std::string>(dd4hep::dd::noNamespace(touch->GetVolume(levels - hfLevels_[it])->GetName())))
              .substr(0, 4);
      if (name == hfNames_[it]) {
        return true;
      }
    }
  }
  return false;
}

bool MaterialBudgetHcalProducer::isItEC(const std::string& name) {
  std::vector<std::string>::const_iterator it = sensitiveEC_.begin();
  std::vector<std::string>::const_iterator itEnd = sensitiveEC_.end();
  for (; it != itEnd; ++it)
    if (name.substr(0, 4) == *it)
      return true;
  return false;
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(MaterialBudgetHcalProducer);
