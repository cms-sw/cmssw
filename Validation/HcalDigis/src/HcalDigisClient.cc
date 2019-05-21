#include <Validation/HcalDigis/interface/HcalDigisClient.h>
// -*- C++ -*-
//
// Package:    HcalDigisClient
// Class:      HcalDigisClient
//
/**\class HcalDigisClient HcalDigisClient.cc Validation/HcalDigisClient/src/HcalDigisClient.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
 */
//
// Original Author:  Ali Fahim,22 R-013,+41227672649,
//         Created:  Wed Mar 23 22:54:28 CET 2011
//
//

// system include files

HcalDigisClient::HcalDigisClient(const edm::ParameterSet& iConfig) {
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "HcalDigisClient.root");
  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
  msm_ = new std::map<std::string, MonitorElement*>();
}

HcalDigisClient::~HcalDigisClient() { delete msm_; }

void HcalDigisClient::runClient(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
  ig.setCurrentFolder(dirName_);
  std::vector<MonitorElement*> hcalMEs;
  // Since out folders are fixed to three, we can just go over these three folders
  // i.e., CaloTowersV/CaloTowersTask, HcalRecHitsV/HcalRecHitTask, NoiseRatesV/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = ig.getSubdirs();
  for (unsigned int i = 0; i < fullPathHLTFolders.size(); i++) {
    ig.setCurrentFolder(fullPathHLTFolders[i]);
    std::vector<std::string> fullSubPathHLTFolders = ig.getSubdirs();
    for (unsigned int j = 0; j < fullSubPathHLTFolders.size(); j++) {
      if (strcmp(fullSubPathHLTFolders[j].c_str(), "HcalDigisV/HcalDigiTask") == 0) {
        hcalMEs = ig.getContents(fullSubPathHLTFolders[j]);
        ig.setCurrentFolder("HcalDigisV/HcalDigiTask");
        if (!HcalDigisEndjob(hcalMEs, "HB", ib))
          edm::LogError("HcalDigisClient") << "Error in HcalDigisEndjob! HB";
        if (!HcalDigisEndjob(hcalMEs, "HE", ib))
          edm::LogError("HcalDigisClient") << "Error in HcalDigisEndjob! HE";
        if (!HcalDigisEndjob(hcalMEs, "HO", ib))
          edm::LogError("HcalDigisClient") << "Error in HcalDigisEndjob! HO";
        if (!HcalDigisEndjob(hcalMEs, "HF", ib))
          edm::LogError("HcalDigisClient") << "Error in HcalDigisEndjob! HF";
      }
    }
  }
}

int HcalDigisClient::HcalDigisEndjob(const std::vector<MonitorElement*>& hcalMEs,
                                     std::string subdet_,
                                     DQMStore::IBooker& ib) {
  using namespace std;
  string strtmp;

  MonitorElement* nevtot(nullptr);

  std::vector<MonitorElement*> ieta_iphi_occupancy_maps;
  std::vector<std::string> depthID;

  // std::cout << " Number of histos " <<     hcalMEs.size() << std::endl;

  for (unsigned int ih = 0; ih < hcalMEs.size(); ih++) {
    if (hcalMEs[ih]->getName() == "nevtot") {
      nevtot = hcalMEs[ih];
      continue;
    }

    //We search the occupancy maps corresponding to this subdetector
    if ((hcalMEs[ih]->getName().find("HcalDigiTask_ieta_iphi_occupancy_map_depth") != std::string::npos) &&
        (hcalMEs[ih]->getName().find(subdet_) != std::string::npos)) {
      ieta_iphi_occupancy_maps.push_back(hcalMEs[ih]);

      std::string start = "depth";
      std::string end = "_H";

      int position = hcalMEs[ih]->getName().find(start) + start.length();
      int length = hcalMEs[ih]->getName().find(end) - position;

      depthID.push_back(hcalMEs[ih]->getName().substr(position, length));

      continue;
    }
  }

  if (hcalMEs.empty()) {
    edm::LogError("HcalDigisClient") << "No nevtot or maps histo found...";
    return 0;
  }
  if (!nevtot) {
    edm::LogError("HcalDigisClient") << "No nevtot histoo found...";
    return 0;
  }
  if (ieta_iphi_occupancy_maps.empty()) {
    edm::LogError("HcalDigisClient") << "No maps histos found...";
    return 0;
  }

  int ev = nevtot->getEntries();

  if (ev <= 0) {
    edm::LogError("HcalDigisClient") << "normalization factor <= 0!";
    return 0;
  }

  float fev = (float)nevtot->getEntries();

  int depths = ieta_iphi_occupancy_maps.size();

  HistLim ietaLim(85, -42.5, 42.5);

  for (int depth = 1; depth <= depths; depth++) {
    strtmp = "HcalDigiTask_occupancy_vs_ieta_depth" + str(depth) + "_" + subdet_;
    book1D(ib, strtmp, ietaLim);
  }

  std::vector<float> sumphi(depths, 0);
  std::vector<float> sumphie(depths, 0);

  float phi_factor;
  float cnorm;
  float enorm;

  for (int depth = 1; depth <= depths; depth++) {
    int nx = ieta_iphi_occupancy_maps[depth - 1]->getNbinsX();
    int ny = ieta_iphi_occupancy_maps[depth - 1]->getNbinsY();

    for (int i = 1; i <= nx; i++) {
      for (int j = 1; j <= ny; j++) {
        // occupancies
        cnorm = ieta_iphi_occupancy_maps[depth - 1]->getBinContent(i, j) / fev;
        enorm = ieta_iphi_occupancy_maps[depth - 1]->getBinError(i, j) / fev;
        ieta_iphi_occupancy_maps[depth - 1]->setBinContent(i, j, cnorm);
        ieta_iphi_occupancy_maps[depth - 1]->setBinError(i, j, enorm);

      }  //for loop over NbinsYU
    }    //for loop over NbinsX
  }      //for loop over the occupancy maps

  for (int i = 1; i <= 82; i++) {
    int ieta = i - 42;  // -41 -1, 0 40
    if (ieta >= 0)
      ieta += 1;  // -41 -1, 1 41  - to make it detector-like

    if (ieta >= -20 && ieta <= 20) {
      phi_factor = 72.;
    } else {
      if (ieta >= 40 || ieta <= -40)
        phi_factor = 18.;
      else
        phi_factor = 36.;
    }

    //zero the sumphi and sumphie vector at the start of each ieta ring
    sumphi.assign(depths, 0);
    sumphie.assign(depths, 0);

    for (int iphi = 1; iphi <= 72; iphi++) {
      for (int depth = 1; depth <= depths; depth++) {
        int binIeta = ieta_iphi_occupancy_maps[depth - 1]->getTH2F()->GetXaxis()->FindBin(ieta);
        int binIphi = ieta_iphi_occupancy_maps[depth - 1]->getTH2F()->GetYaxis()->FindBin(iphi);

        float content = ieta_iphi_occupancy_maps[depth - 1]->getBinContent(binIeta, binIphi);
        float econtent = ieta_iphi_occupancy_maps[depth - 1]->getBinError(binIeta, binIphi);

        sumphi[depth - 1] += content;
        sumphie[depth - 1] += econtent * econtent;

      }  //for loop over depths
    }    //for loop over phi

    //double deta = double(ieta);

    // occupancies vs ieta
    for (int depth = 1; depth <= depths; depth++) {
      strtmp = "HcalDigiTask_occupancy_vs_ieta_depth" + depthID[depth - 1] + "_" + subdet_;
      MonitorElement* ME = msm_->find(strtmp)->second;
      int ietabin = ME->getTH1F()->GetXaxis()->FindBin(float(ieta));

      if (sumphi[depth - 1] > 1.e-30) {
        cnorm = sumphi[depth - 1] / phi_factor;
        enorm = sqrt(sumphie[depth - 1]) / phi_factor;
        ME->setBinContent(ietabin, cnorm);
        ME->setBinError(ietabin, enorm);
      }
    }
  }  // end of i-loop

  return 1;
}

MonitorElement* HcalDigisClient::monitor(std::string name) {
  if (!msm_->count(name))
    return nullptr;
  else
    return msm_->find(name)->second;
}

std::string HcalDigisClient::str(int x) {
  std::stringstream out;
  out << x;
  return out.str();
}

double HcalDigisClient::integralMETH2D(MonitorElement* ME, int i0, int i1, int j0, int j1) {
  double sum(0);
  for (int i = i0; i <= i1; i++) {
    for (int j = j0; j <= j1; j++) {
      sum += ME->getBinContent(i, j);
    }
  }

  return sum;
}

void HcalDigisClient::scaleMETH2D(MonitorElement* ME, double s) {
  int nx = ME->getNbinsX();
  int ny = ME->getNbinsY();

  double content(0);
  double error(0);
  for (int i = 1; i <= nx; i++) {
    for (int j = 1; j <= ny; j++) {
      content = ME->getBinContent(i, j);
      error = ME->getBinError(i, j);
      content *= s;
      error *= s;
      ME->setBinContent(i, j, content);
      ME->setBinError(i, j, error);
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigisClient);
