#include "Validation/CaloTowers/interface/CaloTowersClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

CaloTowersClient::CaloTowersClient(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  debug_ = false;
  verbose_ = false;
  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
}

CaloTowersClient::~CaloTowersClient() {}

void CaloTowersClient::beginJob() {}

void CaloTowersClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  igetter.setCurrentFolder(dirName_);

  if (verbose_)
    std::cout << "\nrunClient" << std::endl;

  std::vector<MonitorElement*> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three folders
  // i.e., CaloTowersV/CaloTowersTask, HcalRecHitsV/HcalRecHitTask, NoiseRatesV/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = igetter.getSubdirs();
  for (unsigned int i = 0; i < fullPathHLTFolders.size(); i++) {
    if (verbose_)
      std::cout << "\nfullPath: " << fullPathHLTFolders[i] << std::endl;
    igetter.setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = igetter.getSubdirs();
    for (unsigned int j = 0; j < fullSubPathHLTFolders.size(); j++) {
      if (verbose_)
        std::cout << "fullSub: " << fullSubPathHLTFolders[j] << std::endl;

      if (strcmp(fullSubPathHLTFolders[j].c_str(), "CaloTowersV/CaloTowersTask") == 0) {
        hcalMEs = igetter.getContents(fullSubPathHLTFolders[j]);
        if (verbose_)
          std::cout << "hltMES size : " << hcalMEs.size() << std::endl;
        if (!CaloTowersEndjob(hcalMEs))
          std::cout << "\nError in CaloTowersEndjob!" << std::endl << std::endl;
      }
    }
  }
}

// called after entering the CaloTowersV/CaloTowersTask directory
// hcalMEs are within that directory
int CaloTowersClient::CaloTowersEndjob(const std::vector<MonitorElement*>& hcalMEs) {
  int useAllHistos = 0;
  MonitorElement* Ntowers_vs_ieta = nullptr;
  MonitorElement *mapEnergy_N = nullptr, *mapEnergy_E = nullptr, *mapEnergy_H = nullptr, *mapEnergy_EH = nullptr;
  MonitorElement *occupancy_map = nullptr, *occupancy_vs_ieta = nullptr;
  for (unsigned int ih = 0; ih < hcalMEs.size(); ih++) {
    if (strcmp(hcalMEs[ih]->getName().c_str(), "Ntowers_per_event_vs_ieta") == 0) {
      Ntowers_vs_ieta = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_Nentries") == 0) {
      mapEnergy_N = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_energy_H") == 0) {
      useAllHistos++;
      mapEnergy_H = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_energy_E") == 0) {
      useAllHistos++;
      mapEnergy_E = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_energy_EH") == 0) {
      useAllHistos++;
      mapEnergy_EH = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_map_occupancy") == 0) {
      occupancy_map = hcalMEs[ih];
    }
    if (strcmp(hcalMEs[ih]->getName().c_str(), "CaloTowersTask_occupancy_vs_ieta") == 0) {
      occupancy_vs_ieta = hcalMEs[ih];
    }
  }
  if (useAllHistos != 0 && useAllHistos != 3)
    return 0;

  double nevent = mapEnergy_N->getEntries();
  if (verbose_)
    std::cout << "nevent : " << nevent << std::endl;

  // mean number of towers per ieta
  int nx = Ntowers_vs_ieta->getNbinsX();
  float cont;
  float econt;
  float fev = float(nevent);

  for (int i = 1; i <= nx; i++) {
    cont = Ntowers_vs_ieta->getBinContent(i) / fev;
    econt = Ntowers_vs_ieta->getBinError(i) / fev;
    Ntowers_vs_ieta->setBinContent(i, cont);
    Ntowers_vs_ieta->setBinError(i, econt);
  }

  // mean energies & occupancies evaluation

  nx = mapEnergy_N->getNbinsX();
  int ny = mapEnergy_N->getNbinsY();
  float cnorm;
  float enorm;
  float phi_factor;

  for (int i = 1; i <= nx; i++) {
    float sumphi = 0.;
    float sumphie = 0.;

    for (int j = 1; j <= ny; j++) {
      // Emean
      cnorm = mapEnergy_N->getBinContent(i, j);
      //Phi histos are not used in the macros
      if (cnorm > 0.000001 && useAllHistos) {
        cont = mapEnergy_E->getBinContent(i, j) / cnorm;
        econt = mapEnergy_E->getBinError(i, j) / cnorm;
        mapEnergy_E->setBinContent(i, j, cont);
        mapEnergy_E->setBinError(i, j, econt);

        cont = mapEnergy_H->getBinContent(i, j) / cnorm;
        econt = mapEnergy_H->getBinError(i, j) / cnorm;
        mapEnergy_H->setBinContent(i, j, cont);
        mapEnergy_H->setBinError(i, j, econt);

        cont = mapEnergy_EH->getBinContent(i, j) / cnorm;
        econt = mapEnergy_EH->getBinError(i, j) / cnorm;
        mapEnergy_EH->setBinContent(i, j, cont);
        mapEnergy_EH->setBinError(i, j, econt);
      }

      // Occupancy (needed for occupancy vs ieta)
      cnorm = occupancy_map->getBinContent(i, j) / fev;
      enorm = occupancy_map->getBinError(i, j) / fev;
      if (cnorm > 1.e-30)
        occupancy_map->setBinContent(i, j, cnorm);

      sumphi += cnorm;
      sumphie += enorm * enorm;

    }  // end of iphy cycle (j)

    //Occupancy vs ieta histo is drawn
    // phi-factor evaluation for occupancy_vs_ieta calculation
    int ieta = i - 42;  // -41,-1,1,41
    if (ieta >= -20 && ieta <= 20) {
      phi_factor = 72.;
    } else {
      if (ieta >= 40 || ieta <= -40) {
        phi_factor = 18.;
      } else
        phi_factor = 36.;
    }

    cnorm = sumphi / phi_factor;
    enorm = sqrt(sumphie) / phi_factor;
    occupancy_vs_ieta->setBinContent(i, cnorm);
    occupancy_vs_ieta->setBinError(i, enorm);

  }  // end of ieta cycle (i)

  return 1;
}

DEFINE_FWK_MODULE(CaloTowersClient);
