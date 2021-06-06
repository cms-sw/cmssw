#include "Validation/Geometry/interface/MaterialBudgetHcalHistos.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DD4hep/Filter.h"

#include <string>
#include <vector>

using namespace geant_units::operators;

MaterialBudgetHcalHistos::MaterialBudgetHcalHistos(const edm::ParameterSet& p) {
  binEta_ = p.getUntrackedParameter<int>("NBinEta", 260);
  binPhi_ = p.getUntrackedParameter<int>("NBinPhi", 180);
  maxEta_ = p.getUntrackedParameter<double>("MaxEta", 5.2);
  etaLow_ = p.getUntrackedParameter<double>("EtaLow", -5.2);
  etaHigh_ = p.getUntrackedParameter<double>("EtaHigh", 5.2);
  fillHistos_ = p.getUntrackedParameter<bool>("FillHisto", true);
  printSum_ = p.getUntrackedParameter<bool>("PrintSummary", false);
  fromdd4hep_ = p.getUntrackedParameter<bool>("Fromdd4hep", false);
  etaMinP_ = p.getUntrackedParameter<double>("EtaMinP", 5.2);
  etaMaxP_ = p.getUntrackedParameter<double>("EtaMaxP", 0.0);
  etaLowMin_ = p.getUntrackedParameter<double>("EtaLowMin", 0.783);
  etaLowMax_ = p.getUntrackedParameter<double>("EtaLowMax", 0.870);
  etaMidMin_ = p.getUntrackedParameter<double>("EtaMidMin", 2.650);
  etaMidMax_ = p.getUntrackedParameter<double>("EtaMidMax", 2.868);
  etaHighMin_ = p.getUntrackedParameter<double>("EtaHighMin", 2.868);
  etaHighMax_ = p.getUntrackedParameter<double>("EtaHighMax", 3.000);
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos: FillHisto : " << fillHistos_ << " PrintSummary "
                                     << printSum_ << " == Eta plot: NX " << binEta_ << " Range " << -maxEta_ << ":"
                                     << maxEta_ << " Phi plot: NX " << binPhi_ << " Range " << -1._pi << ":" << 1._pi
                                     << " (Eta limit " << etaLow_ << ":" << etaHigh_ << ")"
                                     << " Eta range (" << etaLowMin_ << ":" << etaLowMax_ << "), (" << etaMidMin_ << ":"
                                     << etaMidMax_ << "), (" << etaHighMin_ << ":" << etaHighMax_
                                     << ") Debug for eta range " << etaMinP_ << ":" << etaMaxP_ << "  FromDD4Hep "
                                     << fromdd4hep_;
  if (fillHistos_)
    book();
}

void MaterialBudgetHcalHistos::fillBeginJob(const DDCompactView& cpv) {
  constexpr int32_t addLevel = 0;
  if (fillHistos_) {
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
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Names to be tested for " << attribute << " = "
                                           << value << " has " << sensitives_.size() << " elements";
    for (unsigned int i = 0; i < sensitives_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalHistos: sensitives[" << i << "] = " << sensitives_[i];
    attribute = "Volume";
    value = "HF";
    DDSpecificsMatchesValueFilter filter2{DDValue(attribute, value, 0)};
    DDFilteredView fv2(cpv, filter2);
    hfNames_ = getNames(fv2);
    fv2.firstChild();
    DDsvalues_type sv(fv2.mergedSpecifics());
    std::vector<double> temp = getDDDArray("Levels", sv);
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Names to be tested for " << attribute << " = "
                                           << value << " has " << hfNames_.size() << " elements";
    for (unsigned int i = 0; i < hfNames_.size(); i++) {
      int level = static_cast<int>(temp[i]);
      hfLevels_.push_back(level + addLevel);
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalHistos:  HF[" << i << "] = " << hfNames_[i] << " at level " << hfLevels_[i];
    }

    const std::string ecalRO[2] = {"EcalHitsEB", "EcalHitsEE"};
    attribute = "ReadOutName";
    for (int k = 0; k < 2; k++) {
      value = ecalRO[k];
      DDSpecificsMatchesValueFilter filter3{DDValue(attribute, value, 0)};
      DDFilteredView fv3(cpv, filter3);
      std::vector<std::string> senstmp = getNames(fv3);
      edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Names to be tested for " << attribute
                                             << " = " << value << " has " << senstmp.size() << " elements";
      for (unsigned int i = 0; i < senstmp.size(); i++) {
        std::string name = senstmp[i].substr(0, 4);
        if (std::find(sensitiveEC_.begin(), sensitiveEC_.end(), name) == sensitiveEC_.end())
          sensitiveEC_.push_back(name);
      }
    }
    for (unsigned int i = 0; i < sensitiveEC_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalHistos:sensitiveEC[" << i << "] = " << sensitiveEC_[i];
  }
}

void MaterialBudgetHcalHistos::fillBeginJob(const cms::DDCompactView& cpv) {
  constexpr int32_t addLevel = 1;
  if (fillHistos_) {
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
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Names to be tested for " << attribute << " = "
                                           << value << " has " << sensitives_.size() << " elements";
    for (unsigned int i = 0; i < sensitives_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalHistos: sensitives[" << i << "] = " << sensitives_[i];
    attribute = "Volume";
    value = "HF";
    const cms::DDFilter filter2(attribute, value);
    cms::DDFilteredView fv2(cpv, filter2);
    std::vector<int> temp = fv2.get<std::vector<int> >("hf", "Levels");
    hfNames_ = getNames(fv2);
    edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Names to be tested for " << attribute << " = "
                                           << value << " has " << hfNames_.size() << " elements";
    for (unsigned int i = 0; i < hfNames_.size(); i++) {
      hfLevels_.push_back(temp[i] + addLevel);
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalHistos:  HF[" << i << "] = " << hfNames_[i] << " at level " << hfLevels_[i];
    }

    const std::string ecalRO[2] = {"EcalHitsEB", "EcalHitsEE"};
    attribute = "ReadOutName";
    for (int k = 0; k < 2; k++) {
      value = ecalRO[k];
      const cms::DDFilter filter(attribute, value);
      cms::DDFilteredView fv(cpv, filter);
      std::vector<std::string> senstmp = getNames(fv);
      edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Names to be tested for " << attribute
                                             << " = " << value << " has " << senstmp.size() << " elements";
      for (unsigned int i = 0; i < senstmp.size(); i++) {
        std::string name = senstmp[i].substr(0, 4);
        if (std::find(sensitiveEC_.begin(), sensitiveEC_.end(), name) == sensitiveEC_.end())
          sensitiveEC_.push_back(name);
      }
    }
    for (unsigned int i = 0; i < sensitiveEC_.size(); i++)
      edm::LogVerbatim("MaterialBudgetFull")
          << "MaterialBudgetHcalHistos:sensitiveEC[" << i << "] = " << sensitiveEC_[i];
  }
}

void MaterialBudgetHcalHistos::fillStartTrack(const G4Track* aTrack) {
  id_ = layer_ = steps_ = 0;
  radLen_ = intLen_ = stepLen_ = 0;
  nlayHB_ = nlayHE_ = nlayHF_ = nlayHO_ = 0;

  const G4ThreeVector& dir = aTrack->GetMomentum();
  if (dir.theta() != 0) {
    eta_ = dir.eta();
  } else {
    eta_ = -99;
  }
  phi_ = dir.phi();
  double theEnergy = aTrack->GetTotalEnergy();
  int theID = (int)(aTrack->GetDefinition()->GetPDGEncoding());

  if (printSum_) {
    matList_.clear();
    stepLength_.clear();
    radLength_.clear();
    intLength_.clear();
  }

  if ((std::abs(eta_) >= etaMinP_) && (std::abs(eta_) <= etaMaxP_))
    edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos: Track " << aTrack->GetTrackID() << " Code "
                                       << theID << " Energy " << convertUnitsTo(1._GeV, theEnergy) << " GeV; Eta "
                                       << eta_ << " Phi " << convertRadToDeg(phi_) << " PT "
                                       << convertUnitsTo(1._GeV, dir.perp()) << " GeV *****";
}

void MaterialBudgetHcalHistos::fillPerStep(const G4Step* aStep) {
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
      edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos: Step at " << name << " id " << id_ << ":"
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
  if (fillHistos_) {
    edm::LogVerbatim("MaterialBudgetFull")
        << "Volume " << name << ":" << matName << " EC:Sensitive:HF " << isItEC(name) << ":" << isSensitive(name) << ":"
        << isItHF(touch) << " Eta " << abseta << " HC " << ((touch->GetReplicaNumber(1)) / 1000) << ":"
        << ((touch->GetReplicaNumber(0) / 10) % 100 + 3) << " X0 " << (radLen_ + (step / radl)) << " Lambda "
        << (intLen_ + (step / intl));
    ;
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
        edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Det " << det << " Layer " << lay << " Eta "
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
        edm::LogVerbatim("MaterialBudget")
            << "MaterialBudgetHcalHistos: Step at " << name << " calls filHisto with " << (id_ - 1);
      fillHisto(id_ - 1);
    }
  }

  stepLen_ += step;
  radLen_ += (step / radl);
  intLen_ += (step / intl);
  if (fillHistos_) {
    if (id_ == 21) {
      if (!isItHF(aStep->GetPostStepPoint()->GetTouchable())) {
        if ((abseta >= etaMinP_) && (abseta <= etaMaxP_))
          edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos: After HF in " << name << ":"
                                             << static_cast<std::string>(dd4hep::dd::noNamespace(
                                                    aStep->GetPostStepPoint()->GetTouchable()->GetVolume(0)->GetName()))
                                             << " calls fillHisto with " << id_;
        fillHisto(idOld);
        ++id_;
        layer_ = 0;
      }
    }
  }
}

void MaterialBudgetHcalHistos::fillEndTrack() {
  if ((std::abs(eta_) >= etaMinP_) && (std::abs(eta_) <= etaMaxP_))
    edm::LogVerbatim("MaterialBudget") << "Number of layers hit in HB:" << nlayHB_ << " HE:" << nlayHE_
                                       << " HO:" << nlayHO_ << " HF:" << nlayHF_;
  if (fillHistos_) {
    fillHisto(maxSet_ - 1);
    fillLayer();
  }
  if (printSum_) {
    for (unsigned int ii = 0; ii < matList_.size(); ii++) {
      edm::LogVerbatim("MaterialBudget") << matList_[ii] << "\t" << stepLength_[ii] << "\t" << radLength_[ii] << "\t"
                                         << intLength_[ii];
    }
  }
}

void MaterialBudgetHcalHistos::book() {
  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  double maxPhi = 1._pi;
  edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos: Booking user histos === with " << binEta_
                                         << " bins in eta from " << -maxEta_ << " to " << maxEta_ << " and " << binPhi_
                                         << " bins in phi from " << -maxPhi << " to " << maxPhi;

  std::string iter;
  std::string range0 = "(" + std::to_string(etaMidMin_) + ":" + std::to_string(etaMidMax_) + ") ";
  std::string range1 = "(" + std::to_string(etaHighMin_) + ":" + std::to_string(etaHighMax_) + ") ";
  std::string range2 = "(" + std::to_string(etaLowMin_) + ":" + std::to_string(etaLowMax_) + ") ";
  // total X0
  for (int i = 0; i < maxSet_; i++) {
    iter = std::to_string(i);
    me100[i] = tfile->make<TProfile>(
        std::to_string(i + 100).c_str(), ("MB(X0) prof Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me200[i] = tfile->make<TProfile>(
        std::to_string(i + 200).c_str(), ("MB(L0) prof Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me300[i] = tfile->make<TProfile>(
        std::to_string(i + 300).c_str(), ("MB(Step) prof Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me400[i] = tfile->make<TH1F>(
        std::to_string(i + 400).c_str(), ("Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me500[i] = tfile->make<TProfile>(
        std::to_string(i + 500).c_str(), ("MB(X0) prof Ph in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me600[i] = tfile->make<TProfile>(
        std::to_string(i + 600).c_str(), ("MB(L0) prof Ph in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me700[i] = tfile->make<TProfile>(
        std::to_string(i + 700).c_str(), ("MB(Step) prof Ph in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me800[i] =
        tfile->make<TH1F>(std::to_string(i + 800).c_str(), ("Phi in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me900[i] = tfile->make<TProfile2D>(std::to_string(i + 900).c_str(),
                                       ("MB(X0) prof Eta Phi in region " + iter).c_str(),
                                       binEta_ / 2,
                                       -maxEta_,
                                       maxEta_,
                                       binPhi_ / 2,
                                       -maxPhi,
                                       maxPhi);
    me1000[i] = tfile->make<TProfile2D>(std::to_string(i + 1000).c_str(),
                                        ("MB(L0) prof Eta Phi in region " + iter).c_str(),
                                        binEta_ / 2,
                                        -maxEta_,
                                        maxEta_,
                                        binPhi_ / 2,
                                        -maxPhi,
                                        maxPhi);
    me1100[i] = tfile->make<TProfile2D>(std::to_string(i + 1100).c_str(),
                                        ("MB(Step) prof Eta Phi in region " + iter).c_str(),
                                        binEta_ / 2,
                                        -maxEta_,
                                        maxEta_,
                                        binPhi_ / 2,
                                        -maxPhi,
                                        maxPhi);
    me1200[i] = tfile->make<TH2F>(std::to_string(i + 1200).c_str(),
                                  ("Eta vs Phi in region " + iter).c_str(),
                                  binEta_ / 2,
                                  -maxEta_,
                                  maxEta_,
                                  binPhi_ / 2,
                                  -maxPhi,
                                  maxPhi);
    me1600[i] = tfile->make<TProfile>(std::to_string(i + 1600).c_str(),
                                      ("MB(X0) prof Ph in region " + range0 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me1700[i] = tfile->make<TProfile>(std::to_string(i + 1700).c_str(),
                                      ("MB(L0) prof Ph in region " + range0 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me1800[i] = tfile->make<TProfile>(std::to_string(i + 1800).c_str(),
                                      ("MB(Step) prof Ph in region " + range0 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me1900[i] = tfile->make<TProfile>(std::to_string(i + 1900).c_str(),
                                      ("MB(X0) prof Ph in region " + range1 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me2000[i] = tfile->make<TProfile>(std::to_string(i + 2000).c_str(),
                                      ("MB(L0) prof Ph in region " + range1 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me2100[i] = tfile->make<TProfile>(std::to_string(i + 2100).c_str(),
                                      ("MB(Step) prof Ph in region " + range1 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me2200[i] = tfile->make<TProfile>(std::to_string(i + 2200).c_str(),
                                      ("MB(X0) prof Ph in region " + range2 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me2300[i] = tfile->make<TProfile>(std::to_string(i + 2300).c_str(),
                                      ("MB(L0) prof Ph in region " + range2 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
    me2400[i] = tfile->make<TProfile>(std::to_string(i + 2400).c_str(),
                                      ("MB(Step) prof Ph in region " + range2 + iter).c_str(),
                                      binPhi_,
                                      -maxPhi,
                                      maxPhi);
  }
  for (int i = 0; i < maxSet2_; i++) {
    iter = std::to_string(i);
    me1300[i] = tfile->make<TH1F>(std::to_string(i + 1300).c_str(),
                                  ("Events with layers Hit (0 all, 1 HB, ..) for " + iter).c_str(),
                                  binEta_,
                                  -maxEta_,
                                  maxEta_);
    me1400[i] = tfile->make<TH2F>(std::to_string(i + 1400).c_str(),
                                  ("Eta vs Phi for layers hit in " + iter).c_str(),
                                  binEta_ / 2,
                                  -maxEta_,
                                  maxEta_,
                                  binPhi_ / 2,
                                  -maxPhi,
                                  maxPhi);
    me1500[i] = tfile->make<TProfile>(std::to_string(i + 1500).c_str(),
                                      ("Number of layers crossed (0 all, 1 HB, ..) for " + iter).c_str(),
                                      binEta_,
                                      -maxEta_,
                                      maxEta_);
  }

  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos: Booking user histos done ===";
}

void MaterialBudgetHcalHistos::fillHisto(int ii) {
  if ((std::abs(eta_) >= etaMinP_) && (std::abs(eta_) <= etaMaxP_))
    edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos:FillHisto called with index " << ii
                                       << " integrated  step " << stepLen_ << " X0 " << radLen_ << " Lamda " << intLen_;

  if (ii >= 0 && ii < maxSet_) {
    me100[ii]->Fill(eta_, radLen_);
    me200[ii]->Fill(eta_, intLen_);
    me300[ii]->Fill(eta_, stepLen_);
    me400[ii]->Fill(eta_);

    if (eta_ >= etaLow_ && eta_ <= etaHigh_) {
      me500[ii]->Fill(phi_, radLen_);
      me600[ii]->Fill(phi_, intLen_);
      me700[ii]->Fill(phi_, stepLen_);
      me800[ii]->Fill(phi_);
    }

    me900[ii]->Fill(eta_, phi_, radLen_);
    me1000[ii]->Fill(eta_, phi_, intLen_);
    me1100[ii]->Fill(eta_, phi_, stepLen_);
    me1200[ii]->Fill(eta_, phi_);

    if ((std::abs(eta_) >= etaMidMin_) && (std::abs(eta_) <= etaMidMax_)) {
      me1600[ii]->Fill(phi_, radLen_);
      me1700[ii]->Fill(phi_, intLen_);
      me1800[ii]->Fill(phi_, stepLen_);
    }

    if ((std::abs(eta_) >= etaHighMin_) && (std::abs(eta_) <= etaHighMax_)) {
      me1900[ii]->Fill(phi_, radLen_);
      me2000[ii]->Fill(phi_, intLen_);
      me2100[ii]->Fill(phi_, stepLen_);
    }

    if ((std::abs(eta_) >= etaLowMin_) && (std::abs(eta_) <= etaLowMax_)) {
      me2200[ii]->Fill(phi_, radLen_);
      me2300[ii]->Fill(phi_, intLen_);
      me2400[ii]->Fill(phi_, stepLen_);
    }
  }
}

void MaterialBudgetHcalHistos::fillLayer() {
  me1300[0]->Fill(eta_);
  me1400[0]->Fill(eta_, phi_);
  if (nlayHB_ > 0) {
    me1300[1]->Fill(eta_);
    me1400[1]->Fill(eta_, phi_);
  }
  if (nlayHB_ >= 16) {
    me1300[2]->Fill(eta_);
    me1400[2]->Fill(eta_, phi_);
  }
  if (nlayHE_ > 0) {
    me1300[3]->Fill(eta_);
    me1400[3]->Fill(eta_, phi_);
  }
  if (nlayHE_ >= 16) {
    me1300[4]->Fill(eta_);
    me1400[4]->Fill(eta_, phi_);
  }
  if (nlayHO_ > 0) {
    me1300[5]->Fill(eta_);
    me1400[5]->Fill(eta_, phi_);
  }
  if (nlayHO_ >= 2) {
    me1300[6]->Fill(eta_);
    me1400[6]->Fill(eta_, phi_);
  }
  if (nlayHF_ > 0) {
    me1300[7]->Fill(eta_);
    me1400[7]->Fill(eta_, phi_);
  }
  if (nlayHB_ > 0 || nlayHE_ > 0 || (nlayHF_ > 0 && std::abs(eta_) > 3.0)) {
    me1300[8]->Fill(eta_);
    me1400[8]->Fill(eta_, phi_);
  }
  me1500[0]->Fill(eta_, (double)(nlayHB_ + nlayHO_ + nlayHE_ + nlayHF_));
  me1500[1]->Fill(eta_, (double)(nlayHB_));
  me1500[2]->Fill(eta_, (double)(nlayHE_));
  me1500[4]->Fill(eta_, (double)(nlayHF_));
}

void MaterialBudgetHcalHistos::hend() {
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalHistos: Save user histos ===";
}

std::vector<std::string> MaterialBudgetHcalHistos::getNames(DDFilteredView& fv) {
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

std::vector<std::string> MaterialBudgetHcalHistos::getNames(cms::DDFilteredView& fv) {
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

std::vector<double> MaterialBudgetHcalHistos::getDDDArray(const std::string& str, const DDsvalues_type& sv) {
  edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalHistos:getDDDArray called for " << str;
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    edm::LogVerbatim("MaterialBudgetFull") << value;
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      throw cms::Exception("MaterialBudgetHcalHistos") << "nval = " << nval << " < 1 for array " << str << "\n";
    }

    return fvec;
  } else {
    throw cms::Exception("MaterialBudgetHcalHistos") << "cannot get array " << str << "\n";
  }
}

bool MaterialBudgetHcalHistos::isSensitive(const std::string& name) {
  std::vector<std::string>::const_iterator it = sensitives_.begin();
  std::vector<std::string>::const_iterator itEnd = sensitives_.end();
  std::string namx = (name.find('_') == std::string::npos) ? name : name.substr(0, name.find('_'));
  for (; it != itEnd; ++it)
    if (namx == *it)
      return true;
  return false;
}

bool MaterialBudgetHcalHistos::isItHF(const G4VTouchable* touch) {
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

bool MaterialBudgetHcalHistos::isItEC(const std::string& name) {
  std::vector<std::string>::const_iterator it = sensitiveEC_.begin();
  std::vector<std::string>::const_iterator itEnd = sensitiveEC_.end();
  for (; it != itEnd; ++it)
    if (name.substr(0, 4) == *it)
      return true;
  return false;
}
