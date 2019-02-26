#include "Validation/Geometry/interface/MaterialBudgetCastorHistos.h"

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

MaterialBudgetCastorHistos::MaterialBudgetCastorHistos(const edm::ParameterSet &p){

  binEta      = p.getUntrackedParameter<int>("NBinEta", 100);
  binPhi      = p.getUntrackedParameter<int>("NBinPhi", 180);
  etaLow      = p.getUntrackedParameter<double>("EtaLow",  5.0);
  etaHigh     = p.getUntrackedParameter<double>("EtaHigh", 7.0);
  fillHistos  = p.getUntrackedParameter<bool>("FillHisto", true);
  printSum    = p.getUntrackedParameter<bool>("PrintSummary", false);
  edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: FillHisto : "
				 << fillHistos << " PrintSummary " << printSum
				 << " == Eta plot: NX " << binEta << " Range "
				 << etaLow << " (" << -etaHigh << ") : " 
				 << etaHigh << " (" << -etaLow <<") Phi plot: "
				 << "NX " << binPhi << " Range " << -pi << ":"
				 << pi  << " (Eta limit " << etaLow << ":" 
				 << etaHigh <<")";
  if (fillHistos) book();

}

MaterialBudgetCastorHistos::~MaterialBudgetCastorHistos() {
  edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: Save user "
				 << "histos ===";
}

void MaterialBudgetCastorHistos::fillStartTrack(const G4Track* aTrack) {

  id1    = id2    = steps   = 0;
  radLen = intLen = stepLen = 0;

  const G4ThreeVector& dir = aTrack->GetMomentum() ;
  if (dir.theta() != 0 ) {
    eta = dir.eta();
  } else {
    eta = -99;
  }
  phi = dir.phi();
  double theEnergy = aTrack->GetTotalEnergy();
  int    theID     = (int)(aTrack->GetDefinition()->GetPDGEncoding());

  if (printSum) {
    matList.clear();
    stepLength.clear();
    radLength.clear();
    intLength.clear();
  }

  edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: Track " 
				 << aTrack->GetTrackID() << " Code " << theID
				 << " Energy " << theEnergy/GeV << " GeV; Eta "
				 << eta << " Phi " << phi/deg << " PT "
				 << dir.perp()/GeV << " GeV *****";
}


void MaterialBudgetCastorHistos::fillPerStep(const G4Step* aStep) {

  G4Material * material = aStep->GetPreStepPoint()->GetMaterial();
  double step    = aStep->GetStepLength();
  double radl    = material->GetRadlen();
  double intl    = material->GetNuclearInterLength();
  double density = material->GetDensity() / (g/cm3);

  int    id1Old   = id1;
  int    id2Old   = id2;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  std::string         name  = touch->GetVolume(0)->GetName();
  const std::string&         matName = material->GetName();
  if (printSum) {
    bool found = false;
    for (unsigned int ii=0; ii<matList.size(); ii++) {
      if (matList[ii] == matName) {
	stepLength[ii] += step;
	radLength[ii]  += (step/radl);
	intLength[ii]  += (step/intl);
	found           = true;
	break;
      }
    }
    if (!found) {
      matList.push_back(matName);
      stepLength.push_back(step);
      radLength.push_back(step/radl);
      intLength.push_back(step/intl);
    }
    edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: "
				   << name << " " << step << " " << matName 
				   << " " << stepLen << " " << step/radl << " " 
				   << radLen << " " <<step/intl << " " <<intLen;
  } else {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: Step at " 
				   << name << " Length " << step << " in " 
				   << matName << " of density " << density 
				   << " g/cc; Radiation Length " <<radl <<" mm;"
				   << " Interaction Length " << intl << " mm\n"
				   << "                          Position " 
				   << aStep->GetPreStepPoint()->GetPosition()
				   << " Cylindrical R "
				   <<aStep->GetPreStepPoint()->GetPosition().perp()
				   << " Length (so far) " << stepLen << " L/X0 "
				   << step/radl << "/" << radLen << " L/Lambda "
				   << step/intl << "/" << intLen;
  }

  int level = ((touch->GetHistoryDepth())+1);
  std::string name1="XXXX", name2="XXXX";
  if (level>3) name1 = touch->GetVolume(level-4)->GetName();
  if (level>4) name2 = touch->GetVolume(level-5)->GetName();
  if (name1 == "CAST") {
    id1 = 1;
    if      (name2 == "CAEC") id2 = 2;
    else if (name2 == "CAHC") id2 = 3;
    else if (name2 == "CEDC") id2 = 4;
    else if (name2 == "CHDC") id2 = 5;
    else                      id2 = 0;
  } else {
    id1 = id2 = 0;
  }
  LogDebug("MaterialBudget") << "MaterialBudgetCastorHistos: Level " << level
			     << " Volume " << name1 << " and " << name2
			     << " ID1 " << id1 << " (" << id1Old << ") ID2 "
			     << id2 << " (" << id2Old << ")";

  if (fillHistos) {
    if (id1 != id1Old) {
      if (id1 == 0) fillHisto(id1Old, 1);
      else          fillHisto(id1,    0);
    }
    if (id2 != id2Old) {
      if (id2 == 0) fillHisto(id2Old, 1);
      else          fillHisto(id2,    0);
    }
  }

  stepLen += step;
  radLen  += step/radl;
  intLen  += step/intl;
}


void MaterialBudgetCastorHistos::fillEndTrack() {

  if (fillHistos) {
    if (id1 != 0) fillHisto(id1, 1);
    if (id2 != 0) fillHisto(id2, 1);
  }
  if (printSum) {
    for (unsigned int ii=0; ii<matList.size(); ii++) {
      edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: "
				     << matList[ii] << "\t" << stepLength[ii]
				     << "\t" << radLength[ii] << "\t"
				     << intLength[ii];
    }
  }
}

void MaterialBudgetCastorHistos::book() {

  // Book histograms
  edm::Service<TFileService> tfile;
  
  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "MaterialBudgetCastorHistos: TFileService unavailable: "
                                      << "please add it to config file";

  double maxPhi=pi;
  edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: Booking user "
				 << "histos === with " << binEta << " bins "
				 << "in eta from " << etaLow << " to "
				 << etaHigh << " and " << binPhi << " bins "
				 << "in phi from " << -maxPhi << " to " 
				 << maxPhi;
  
  std::string tag1, tag2;
  // total X0
  for (int i=0; i<maxSet; i++) {
    double minEta=etaLow;
    double maxEta=etaHigh;
    int    ireg = i;
    if (i > 9) {
      minEta = -etaHigh;
      maxEta = -etaLow;
      ireg  -= 10;
      tag2 = " (-ve Eta Side)";
    } else {
      tag2 = " (+ve Eta Side)";
    }
    if ((i%2) == 0) {
      ireg  /= 2;
      tag1 = " == Start";
    } else {
      ireg   = (ireg-1)/2;
      tag1 = " == End";
    }
    std::string title = std::to_string(ireg) + tag1 + tag2;
    me100[i] = tfile->make<TProfile>(std::to_string(i + 100).c_str(),
                  ("MB(X0) prof Eta in region " + title).c_str(), binEta, minEta, maxEta);
    me200[i] = tfile->make<TProfile>(std::to_string(i + 200).c_str(),
                  ("MB(L0) prof Eta in region " + title).c_str(), binEta, minEta, maxEta);
    me300[i] = tfile->make<TProfile>(std::to_string(i + 300).c_str(),
                  ("MB(Step) prof Eta in region " + title).c_str(), binEta, minEta, maxEta);
    me400[i] = tfile->make<TH1F>(std::to_string(i + 400).c_str(),
                  ("Eta in region " + title).c_str(), binEta, minEta, maxEta);
    me500[i] = tfile->make<TProfile>(std::to_string(i + 500).c_str(),
                  ("MB(X0) prof Ph in region " + title).c_str(), binPhi, -maxPhi, maxPhi);
    me600[i] = tfile->make<TProfile>(std::to_string(i + 600).c_str(),
                  ("MB(L0) prof Ph in region " + title).c_str(), binPhi, -maxPhi, maxPhi);
    me700[i] = tfile->make<TProfile>(std::to_string(i + 700).c_str(),
                  ("MB(Step) prof Ph in region " + title).c_str(), binPhi, -maxPhi, maxPhi);
    me800[i] = tfile->make<TH1F>(std::to_string(i + 800).c_str(),
                  ("Phi in region " + title).c_str(), binPhi, -maxPhi, maxPhi);
    me900[i] = tfile->make<TProfile2D>(std::to_string(i + 900).c_str(),
                  ("MB(X0) prof Eta Phi in region " + title).c_str(), binEta/2, minEta, maxEta,
                  binPhi/2, -maxPhi, maxPhi);
    me1000[i]= tfile->make<TProfile2D>(std::to_string(i + 1000).c_str(),
                  ("MB(L0) prof Eta Phi in region " + title).c_str(), binEta/2, minEta, maxEta,
                  binPhi/2, -maxPhi, maxPhi);
    me1100[i]= tfile->make<TProfile2D>(std::to_string(i + 1100).c_str(),
                  ("MB(Step) prof Eta Phi in region " + title).c_str(), binEta/2, minEta, maxEta,
                  binPhi/2, -maxPhi, maxPhi);
    me1200[i]= tfile->make<TH2F>(std::to_string(i + 1200).c_str(),
                  ("Eta vs Phi in region " + title).c_str(), binEta/2, minEta, maxEta,
                  binPhi/2, -maxPhi, maxPhi);
  }

  edm::LogInfo("MaterialBudget") << "MaterialBudgetCastorHistos: Booking user "
				 << "histos done ===";

}

void MaterialBudgetCastorHistos::fillHisto(int id, int ix) {

  int ii = 2*(id-1) + ix;
  double etaAbs = eta;
  if (eta < 0) {
    etaAbs = -eta;
    ii    += 10;
  }
  LogDebug("MaterialBudget") << "MaterialBudgetCastorHistos:FillHisto "
			     << "called with index " << id << ":" << ix 
			     << ":" << ii << " eta " << etaAbs << " (" 
			     << eta << ") integrated  step " << stepLen 
			     << " X0 " << radLen << " Lamda " << intLen;
  
  me100[ii]->Fill(eta, radLen);
  me200[ii]->Fill(eta, intLen);
  me300[ii]->Fill(eta, stepLen);
  me400[ii]->Fill(eta);

  if (etaAbs >= etaLow && etaAbs <= etaHigh) {
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
