#include "Validation/Geometry/interface/MaterialBudgetHcalHistos.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <string>

MaterialBudgetHcalHistos::MaterialBudgetHcalHistos(const edm::ParameterSet& p) {
  binEta = p.getUntrackedParameter<int>("NBinEta", 260);
  binPhi = p.getUntrackedParameter<int>("NBinPhi", 180);
  maxEta = p.getUntrackedParameter<double>("MaxEta", 5.2);
  etaLow = p.getUntrackedParameter<double>("EtaLow", -5.2);
  etaHigh = p.getUntrackedParameter<double>("EtaHigh", 5.2);
  fillHistos = p.getUntrackedParameter<bool>("FillHisto", true);
  printSum = p.getUntrackedParameter<bool>("PrintSummary", false);
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: FillHisto : " << fillHistos << " PrintSummary "
                                 << printSum << " == Eta plot: NX " << binEta << " Range " << -maxEta << ":" << maxEta
                                 << " Phi plot: NX " << binPhi << " Range " << -pi << ":" << pi << " (Eta limit "
                                 << etaLow << ":" << etaHigh << ")";
  if (fillHistos)
    book();
}

void MaterialBudgetHcalHistos::fillBeginJob(const DDCompactView& cpv) {
  if (fillHistos) {
    std::string attribute = "ReadOutName";
    std::string value = "HcalHits";
    DDSpecificsMatchesValueFilter filter1{DDValue(attribute, value, 0)};
    DDFilteredView fv1(cpv, filter1);
    sensitives = getNames(fv1);
    edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Names to be "
                                   << "tested for " << attribute << " = " << value << " has " << sensitives.size()
                                   << " elements";
    for (unsigned int i = 0; i < sensitives.size(); i++)
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: sensitives[" << i << "] = " << sensitives[i];

    attribute = "Volume";
    value = "HF";
    DDSpecificsMatchesValueFilter filter2{DDValue(attribute, value, 0)};
    DDFilteredView fv2(cpv, filter2);
    hfNames = getNames(fv2);
    fv2.firstChild();
    DDsvalues_type sv(fv2.mergedSpecifics());
    std::vector<double> temp = getDDDArray("Levels", sv);
    edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Names to be "
                                   << "tested for " << attribute << " = " << value << " has " << hfNames.size()
                                   << " elements";
    for (unsigned int i = 0; i < hfNames.size(); i++) {
      int level = static_cast<int>(temp[i]);
      hfLevels.push_back(level);
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos:  HF[" << i << "] = " << hfNames[i] << " at level "
                                     << hfLevels[i];
    }

    std::string ecalRO[2] = {"EcalHitsEB", "EcalHitsEE"};
    attribute = "ReadOutName";
    for (int k = 0; k < 2; k++) {
      value = ecalRO[k];
      DDSpecificsMatchesValueFilter filter3{DDValue(attribute, value, 0)};
      DDFilteredView fv3(cpv, filter3);
      std::vector<std::string> senstmp = getNames(fv3);
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Names to be"
                                     << " tested for " << attribute << " = " << value << " has " << senstmp.size()
                                     << " elements";
      for (unsigned int i = 0; i < senstmp.size(); i++)
        sensitiveEC.push_back(senstmp[i]);
    }
    for (unsigned int i = 0; i < sensitiveEC.size(); i++)
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos:sensitiveEC[" << i << "] = " << sensitiveEC[i];
  }
}

void MaterialBudgetHcalHistos::fillStartTrack(const G4Track* aTrack) {
  id = layer = steps = 0;
  radLen = intLen = stepLen = 0;
  nlayHB = nlayHE = nlayHF = nlayHO = 0;

  const G4ThreeVector& dir = aTrack->GetMomentum();
  if (dir.theta() != 0) {
    eta = dir.eta();
  } else {
    eta = -99;
  }
  phi = dir.phi();
  double theEnergy = aTrack->GetTotalEnergy();
  int theID = (int)(aTrack->GetDefinition()->GetPDGEncoding());

  if (printSum) {
    matList.clear();
    stepLength.clear();
    radLength.clear();
    intLength.clear();
  }

  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Track " << aTrack->GetTrackID() << " Code " << theID
                                 << " Energy " << theEnergy / GeV << " GeV; Eta " << eta << " Phi " << phi / deg
                                 << " PT " << dir.perp() / GeV << " GeV *****";
}

void MaterialBudgetHcalHistos::fillPerStep(const G4Step* aStep) {
  G4Material* material = aStep->GetPreStepPoint()->GetMaterial();
  double step = aStep->GetStepLength();
  double radl = material->GetRadlen();
  double intl = material->GetNuclearInterLength();
  double density = material->GetDensity() / (g / cm3);

  int idOld = id;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  std::string name = touch->GetVolume(0)->GetName();
  const std::string& matName = material->GetName();
  if (printSum) {
    bool found = false;
    for (unsigned int ii = 0; ii < matList.size(); ii++) {
      if (matList[ii] == matName) {
        stepLength[ii] += step;
        radLength[ii] += (step / radl);
        intLength[ii] += (step / intl);
        found = true;
        break;
      }
    }
    if (!found) {
      matList.push_back(matName);
      stepLength.push_back(step);
      radLength.push_back(step / radl);
      intLength.push_back(step / intl);
    }
    edm::LogInfo("MaterialBudget") << name << " " << step << " " << matName << " " << stepLen << " " << step / radl
                                   << " " << radLen << " " << step / intl << " " << intLen;
  } else {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Step at " << name << " Length " << step << " in "
                                   << matName << " of density " << density << " g/cc; Radiation Length " << radl
                                   << " mm;"
                                   << " Interaction Length " << intl << " mm\n"
                                   << "                          Position " << aStep->GetPreStepPoint()->GetPosition()
                                   << " Cylindrical R " << aStep->GetPreStepPoint()->GetPosition().perp()
                                   << " Length (so far) " << stepLen << " L/X0 " << step / radl << "/" << radLen
                                   << " L/Lambda " << step / intl << "/" << intLen;
  }

  int det = 0, lay = 0;
  if (fillHistos) {
    if (isItEC(name)) {
      det = 1;
      lay = 1;
    } else {
      if (isSensitive(name)) {
        if (isItHF(touch)) {
          det = 5;
          lay = 21;
          if (lay != layer)
            nlayHF++;
        } else {
          det = (touch->GetReplicaNumber(1)) / 1000;
          lay = (touch->GetReplicaNumber(0) / 10) % 100 + 3;
          if (det == 4) {
            double abeta = std::abs(eta);
            if (abeta < 1.479)
              lay = layer + 1;
            else
              lay--;
            if (lay < 3)
              lay = 3;
            if (lay == layer)
              lay++;
            if (lay > 20)
              lay = 20;
            if (lay != layer)
              nlayHE++;
          } else if (lay != layer) {
            if (lay >= 20)
              nlayHO++;
            else
              nlayHB++;
          }
        }
        LogDebug("MaterialBudget") << "MaterialBudgetHcalHistos: Det " << det << " Layer " << lay << " Eta " << eta
                                   << " Phi " << phi / deg;
      } else if (layer == 1) {
        det = -1;
        lay = 2;
      }
    }
    if (det != 0) {
      if (lay != layer) {
        id = lay;
        layer = lay;
      }
    }

    if (id > idOld) {
      //    edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Step at " << name;
      fillHisto(id - 1);
    }
  }

  stepLen += step;
  radLen += step / radl;
  intLen += step / intl;
  if (fillHistos) {
    if (layer == 21 && det == 5) {
      if (!isItHF(aStep->GetPostStepPoint()->GetTouchable())) {
        LogDebug("MaterialBudget") << "MaterialBudgetHcalHistos: After HF in "
                                   << aStep->GetPostStepPoint()->GetTouchable()->GetVolume(0)->GetName();
        fillHisto(id);
        id++;
        layer = 0;
      }
    }
  }
}

void MaterialBudgetHcalHistos::fillEndTrack() {
  edm::LogInfo("MaterialBudget") << "Number of layers hit in HB:" << nlayHB << " HE:" << nlayHE << " HO:" << nlayHO
                                 << " HF:" << nlayHF;
  if (fillHistos) {
    fillHisto(maxSet - 1);
    fillLayer();
  }
  if (printSum) {
    for (unsigned int ii = 0; ii < matList.size(); ii++) {
      edm::LogInfo("MaterialBudget") << matList[ii] << "\t" << stepLength[ii] << "\t" << radLength[ii] << "\t"
                                     << intLength[ii];
    }
  }
}

void MaterialBudgetHcalHistos::book() {
  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  double maxPhi = pi;
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Booking user "
                                 << "histos === with " << binEta << " bins "
                                 << "in eta from " << -maxEta << " to " << maxEta << " and " << binPhi << " bins "
                                 << "in phi from " << -maxPhi << " to " << maxPhi;

  std::string iter;
  // total X0
  for (int i = 0; i < maxSet; i++) {
    iter = std::to_string(i);
    me100[i] = tfile->make<TProfile>(
        std::to_string(i + 100).c_str(), ("MB(X0) prof Eta in region " + iter).c_str(), binEta, -maxEta, maxEta);
    me200[i] = tfile->make<TProfile>(
        std::to_string(i + 200).c_str(), ("MB(L0) prof Eta in region " + iter).c_str(), binEta, -maxEta, maxEta);
    me300[i] = tfile->make<TProfile>(
        std::to_string(i + 300).c_str(), ("MB(Step) prof Eta in region " + iter).c_str(), binEta, -maxEta, maxEta);
    me400[i] =
        tfile->make<TH1F>(std::to_string(i + 400).c_str(), ("Eta in region " + iter).c_str(), binEta, -maxEta, maxEta);
    me500[i] = tfile->make<TProfile>(
        std::to_string(i + 500).c_str(), ("MB(X0) prof Ph in region " + iter).c_str(), binPhi, -maxPhi, maxPhi);
    me600[i] = tfile->make<TProfile>(
        std::to_string(i + 600).c_str(), ("MB(L0) prof Ph in region " + iter).c_str(), binPhi, -maxPhi, maxPhi);
    me700[i] = tfile->make<TProfile>(
        std::to_string(i + 700).c_str(), ("MB(Step) prof Ph in region " + iter).c_str(), binPhi, -maxPhi, maxPhi);
    me800[i] =
        tfile->make<TH1F>(std::to_string(i + 800).c_str(), ("Phi in region " + iter).c_str(), binPhi, -maxPhi, maxPhi);
    me900[i] = tfile->make<TProfile2D>(std::to_string(i + 900).c_str(),
                                       ("MB(X0) prof Eta Phi in region " + iter).c_str(),
                                       binEta / 2,
                                       -maxEta,
                                       maxEta,
                                       binPhi / 2,
                                       -maxPhi,
                                       maxPhi);
    me1000[i] = tfile->make<TProfile2D>(std::to_string(i + 1000).c_str(),
                                        ("MB(L0) prof Eta Phi in region " + iter).c_str(),
                                        binEta / 2,
                                        -maxEta,
                                        maxEta,
                                        binPhi / 2,
                                        -maxPhi,
                                        maxPhi);
    me1100[i] = tfile->make<TProfile2D>(std::to_string(i + 1100).c_str(),
                                        ("MB(Step) prof Eta Phi in region " + iter).c_str(),
                                        binEta / 2,
                                        -maxEta,
                                        maxEta,
                                        binPhi / 2,
                                        -maxPhi,
                                        maxPhi);
    me1200[i] = tfile->make<TH2F>(std::to_string(i + 1200).c_str(),
                                  ("Eta vs Phi in region " + iter).c_str(),
                                  binEta / 2,
                                  -maxEta,
                                  maxEta,
                                  binPhi / 2,
                                  -maxPhi,
                                  maxPhi);
  }
  for (int i = 0; i < maxSet2; i++) {
    iter = std::to_string(i);
    me1300[i] = tfile->make<TH1F>(std::to_string(i + 1300).c_str(),
                                  ("Events with layers Hit (0 all, 1 HB, ..) for " + iter).c_str(),
                                  binEta,
                                  -maxEta,
                                  maxEta);
    me1400[i] = tfile->make<TH2F>(std::to_string(i + 1400).c_str(),
                                  ("Eta vs Phi for layers hit in " + iter).c_str(),
                                  binEta / 2,
                                  -maxEta,
                                  maxEta,
                                  binPhi / 2,
                                  -maxPhi,
                                  maxPhi);
    me1500[i] = tfile->make<TProfile>(std::to_string(i + 1500).c_str(),
                                      ("Number of layers crossed (0 all, 1 HB, ..) for " + iter).c_str(),
                                      binEta,
                                      -maxEta,
                                      maxEta);
  }

  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Booking user "
                                 << "histos done ===";
}

void MaterialBudgetHcalHistos::fillHisto(int ii) {
  LogDebug("MaterialBudget") << "MaterialBudgetHcalHistos:FillHisto called "
                             << "with index " << ii << " integrated  step " << stepLen << " X0 " << radLen << " Lamda "
                             << intLen;

  if (ii >= 0 && ii < maxSet) {
    me100[ii]->Fill(eta, radLen);
    me200[ii]->Fill(eta, intLen);
    me300[ii]->Fill(eta, stepLen);
    me400[ii]->Fill(eta);

    if (eta >= etaLow && eta <= etaHigh) {
      me500[ii]->Fill(phi, radLen);
      me600[ii]->Fill(phi, intLen);
      me700[ii]->Fill(phi, stepLen);
      me800[ii]->Fill(phi);
    }

    me900[ii]->Fill(eta, phi, radLen);
    me1000[ii]->Fill(eta, phi, intLen);
    me1100[ii]->Fill(eta, phi, stepLen);
    me1200[ii]->Fill(eta, phi);
  }
}

void MaterialBudgetHcalHistos::fillLayer() {
  me1300[0]->Fill(eta);
  me1400[0]->Fill(eta, phi);
  if (nlayHB > 0) {
    me1300[1]->Fill(eta);
    me1400[1]->Fill(eta, phi);
  }
  if (nlayHB >= 16) {
    me1300[2]->Fill(eta);
    me1400[2]->Fill(eta, phi);
  }
  if (nlayHE > 0) {
    me1300[3]->Fill(eta);
    me1400[3]->Fill(eta, phi);
  }
  if (nlayHE >= 16) {
    me1300[4]->Fill(eta);
    me1400[4]->Fill(eta, phi);
  }
  if (nlayHO > 0) {
    me1300[5]->Fill(eta);
    me1400[5]->Fill(eta, phi);
  }
  if (nlayHO >= 2) {
    me1300[6]->Fill(eta);
    me1400[6]->Fill(eta, phi);
  }
  if (nlayHF > 0) {
    me1300[7]->Fill(eta);
    me1400[7]->Fill(eta, phi);
  }
  if (nlayHB > 0 || nlayHE > 0 || (nlayHF > 0 && std::abs(eta) > 3.0)) {
    me1300[8]->Fill(eta);
    me1400[8]->Fill(eta, phi);
  }
  me1500[0]->Fill(eta, (double)(nlayHB + nlayHO + nlayHE + nlayHF));
  me1500[1]->Fill(eta, (double)(nlayHB));
  me1500[2]->Fill(eta, (double)(nlayHE));
  me1500[4]->Fill(eta, (double)(nlayHF));
}

void MaterialBudgetHcalHistos::hend() {
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHcalHistos: Save user "
                                 << "histos ===";
}

std::vector<std::string> MaterialBudgetHcalHistos::getNames(DDFilteredView& fv) {
  std::vector<std::string> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart& log = fv.logicalPart();
    std::string namx = log.name().name();
    bool ok = true;
    for (unsigned int i = 0; i < tmp.size(); i++)
      if (namx == tmp[i])
        ok = false;
    if (ok)
      tmp.push_back(namx);
    dodet = fv.next();
  }
  return tmp;
}

std::vector<double> MaterialBudgetHcalHistos::getDDDArray(const std::string& str, const DDsvalues_type& sv) {
  LogDebug("MaterialBudget") << "MaterialBudgetHcalHistos:getDDDArray called "
                             << "for " << str;
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    LogDebug("MaterialBudget") << value;
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("MaterialBudget") << "MaterialBudgetHcalHistos : # of " << str << " bins " << nval
                                      << " < 1 ==> illegal";
      throw cms::Exception("Unknown", "MaterialBudgetHcalHistos") << "nval < 1 for array " << str << "\n";
    }

    return fvec;
  } else {
    edm::LogError("MaterialBudget") << "MaterialBudgetHcalHistos : cannot get "
                                    << "array " << str;
    throw cms::Exception("Unknown", "MaterialBudgetHcalHistos") << "cannot get array " << str << "\n";
  }
}

bool MaterialBudgetHcalHistos::isSensitive(std::string name) {
  std::vector<std::string>::const_iterator it = sensitives.begin();
  std::vector<std::string>::const_iterator itEnd = sensitives.end();
  for (; it != itEnd; ++it)
    if (name == *it)
      return true;
  return false;
}

bool MaterialBudgetHcalHistos::isItHF(const G4VTouchable* touch) {
  // std::vector<std::string>::const_iterator it = hfNames.begin();
  int levels = ((touch->GetHistoryDepth()) + 1);
  for (unsigned int it = 0; it < hfNames.size(); it++) {
    if (levels >= hfLevels[it]) {
      std::string name = touch->GetVolume(levels - hfLevels[it])->GetName();
      if (name == hfNames[it])
        return true;
    }
  }
  return false;
}

bool MaterialBudgetHcalHistos::isItEC(std::string name) {
  std::vector<std::string>::const_iterator it = sensitiveEC.begin();
  std::vector<std::string>::const_iterator itEnd = sensitiveEC.end();
  for (; it != itEnd; ++it)
    if (name == *it)
      return true;
  return false;
}
