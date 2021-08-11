#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

#include <string>
#include <vector>

class HFShowerLibraryAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  HFShowerLibraryAnalyzer(const edm::ParameterSet& ps);
  ~HFShowerLibraryAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  void bookHistos();
  void getRecord(int, int);
  void loadEventInfo(TBranch*);

  TFile* hf_;
  TBranch *emBranch_, *hadBranch_;
  bool verbose_, newForm_, v3version_;
  int nMomBin_, totEvents_, evtPerBin_;
  float libVers_, listVersion_;
  std::vector<double> pmom_;
  HFShowerPhotonCollection* photo_;
  HFShowerPhotonCollection photon_;
  std::vector<TH1F*> h_x_[2], h_y_[2], h_z_[2], h_t_[2], h_l_[2], h_nl_[2], h_ns_[2];
};

HFShowerLibraryAnalyzer::HFShowerLibraryAnalyzer(edm::ParameterSet const& ps)
    : hf_(nullptr), emBranch_(nullptr), hadBranch_(nullptr) {
  usesResource(TFileService::kSharedResource);

  edm::FileInPath fp("SimG4CMS/Calo/data/" + ps.getParameter<std::string>("FileName"));
  std::string pTreeName = fp.fullPath();
  std::string emName = ps.getParameter<std::string>("TreeEMID");
  std::string hadName = ps.getParameter<std::string>("TreeHadID");
  std::string branchEvInfo = ps.getParameter<std::string>("BranchEvt");
  std::string branchPre = ps.getParameter<std::string>("BranchPre");
  std::string branchPost = ps.getParameter<std::string>("BranchPost");
  verbose_ = ps.getParameter<bool>("Verbosity");
  evtPerBin_ = ps.getParameter<bool>("EventPerBin");

  if (pTreeName.find('.') == 0)
    pTreeName.erase(0, 2);
  const char* nTree = pTreeName.c_str();
  hf_ = TFile::Open(nTree);

  if (hf_->IsOpen()) {
    edm::LogVerbatim("HFShower") << "HFShowerLibrary: opening " << nTree << " successfully";
    newForm_ = (branchEvInfo.empty());
    TTree* event = ((newForm_) ? static_cast<TTree*>(hf_->Get("HFSimHits")) : static_cast<TTree*>(hf_->Get("Events")));
    if (event) {
      TBranch* evtInfo(nullptr);
      if (!newForm_) {
        std::string info = branchEvInfo + branchPost;
        evtInfo = event->GetBranch(info.c_str());
      }
      if (evtInfo || newForm_) {
        loadEventInfo(evtInfo);
      } else {
        edm::LogError("HFShower") << "HFShowerLibrayEventInfo Branch does not exist in Event";
        throw cms::Exception("Unknown", "HFShowerLibraryAnalyzer") << "Event information absent\n";
      }
    } else {
      edm::LogError("HFShower") << "Events Tree does not exist";
      throw cms::Exception("Unknown", "HFShowerLibraryAnalyzer") << "Events tree absent\n";
    }

    std::stringstream ss;
    ss << "HFShowerLibraryAnalyzer: Library " << libVers_ << " ListVersion " << listVersion_ << " Events Total "
       << totEvents_ << " and " << evtPerBin_ << " per bin\n";
    ss << "HFShowerLibraryAnalyzer: Energies (GeV) with " << nMomBin_ << " bins\n";
    for (int i = 0; i < nMomBin_; ++i) {
      if (i / 10 * 10 == i && i > 0) {
        ss << "\n";
      }
      ss << "  " << pmom_[i] / CLHEP::GeV;
    }
    edm::LogVerbatim("HFShower") << ss.str();

    std::string nameBr = branchPre + emName + branchPost;
    emBranch_ = event->GetBranch(nameBr.c_str());
    if (verbose_)
      emBranch_->Print();
    nameBr = branchPre + hadName + branchPost;
    hadBranch_ = event->GetBranch(nameBr.c_str());
    if (verbose_)
      hadBranch_->Print();

    v3version_ = (emBranch_->GetClassName() == std::string("vector<float>")) ? true : false;
    edm::LogVerbatim("HFShower")
        << " HFShowerLibraryAnalyzer:Branch " << emName << " has " << emBranch_->GetEntries() << " entries and Branch "
        << hadName << " has " << hadBranch_->GetEntries()
        << " entries\n HFShowerLibraryAnalyzer::No packing information - Assume x, y, z are not in packed form";
    photo_ = new HFShowerPhotonCollection;

    bookHistos();
  } else {
    edm::LogError("HFShower") << "HFShowerLibrary: opening " << nTree << " failed";
  }
}

HFShowerLibraryAnalyzer::~HFShowerLibraryAnalyzer() {
  if (hf_)
    hf_->Close();
  delete photo_;
}

void HFShowerLibraryAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FileName", "HFShowerLibrary_10000.root");
  desc.add<std::string>("TreeEMID", "emParticles");
  desc.add<std::string>("TreeHadID", "hadParticles");
  desc.add<std::string>("BranchEvt", "");
  desc.add<std::string>("BranchPre", "");
  desc.add<std::string>("BranchPost", "");
  desc.add<bool>("Verbosity", false);
  desc.add<int>("EventPerBin", 10000);
  descriptions.add("hfShowerLibaryAnalysis", desc);
}

void HFShowerLibraryAnalyzer::bookHistos() {
  edm::Service<TFileService> tfile;
  if (!tfile.isAvailable())
    throw cms::Exception("Unknown", "HFShowerLibraryAnalyzer")
        << "TFileService unavailable: please add it to config file";
  char name[20], title[40], titlx[120];
  TH1F* hist;
  for (int i = 0; i < 2; ++i) {
    std::string type = (i == 0) ? "EM" : "Hadron";
    for (int k = 0; k < nMomBin_; ++k) {
      sprintf(title, "Showers for p = %6.1f GeV", pmom_[k] / CLHEP::GeV);
      sprintf(name, "X%d%d", k, i);
      sprintf(titlx, "x coordinate of %s shower (mm)", type.c_str());
      hist = tfile->make<TH1F>(name, title, 200, -500.0, 500.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      hist->Sumw2();
      h_x_[i].emplace_back(hist);
      sprintf(name, "Y%d%d", k, i);
      sprintf(titlx, "y coordinate of %s shower (mm)", type.c_str());
      hist = tfile->make<TH1F>(name, title, 200, -500.0, 500.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      hist->Sumw2();
      h_y_[i].emplace_back(hist);
      sprintf(name, "Z%d%d", k, i);
      sprintf(titlx, "z coordinate of %s shower (mm)", type.c_str());
      hist = tfile->make<TH1F>(name, title, 200, -2000.0, 2000.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      hist->Sumw2();
      h_z_[i].emplace_back(hist);
      sprintf(name, "T%d%d", k, i);
      sprintf(titlx, "Time of %s shower (ns)", type.c_str());
      hist = tfile->make<TH1F>(name, title, 200, 0.0, 50.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      hist->Sumw2();
      h_t_[i].emplace_back(hist);
      sprintf(name, "L%d%d", k, i);
      sprintf(titlx, "Lambda of %s shower photon (nm)", type.c_str());
      hist = tfile->make<TH1F>(name, title, 200, 300.0, 800.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      h_l_[i].emplace_back(hist);
      hist->Sumw2();
      sprintf(name, "NL%d%d", k, i);
      sprintf(titlx, "Number of %s shower photons in long fibers", type.c_str());
      hist = tfile->make<TH1F>(name, title, 3000, 0.0, 3000.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      h_nl_[i].emplace_back(hist);
      sprintf(name, "NS%d%d", k, i);
      sprintf(titlx, "Number of %s shower photons in short fibes", type.c_str());
      hist = tfile->make<TH1F>(name, title, 3000, 0.0, 3000.0);
      hist->GetXaxis()->SetTitle(titlx);
      hist->GetYaxis()->SetTitle("Shower");
      hist->GetYaxis()->SetTitleOffset(1.2);
      h_ns_[i].emplace_back(hist);
    }
  }

  // Now fill them up
  for (int type = 0; type < 2; ++type) {
    for (int k = 0; k < nMomBin_; ++k) {
      for (int j = 0; j < evtPerBin_; ++j) {
        int irc = k * evtPerBin_ + j + 1;
        getRecord(type, irc);
        int nPhoton = (newForm_) ? photo_->size() : photon_.size();
        int nlong = 0, nshort = 0;
        for (int i = 0; i < nPhoton; i++) {
          if (newForm_) {
            if (photo_->at(i).z() > 0) {
              nlong++;
            } else {
              nshort++;
            }
            h_x_[type][k]->Fill((photo_->at(i)).x());
            h_y_[type][k]->Fill((photo_->at(i)).y());
            h_z_[type][k]->Fill((photo_->at(i)).z());
            h_t_[type][k]->Fill((photo_->at(i)).t());
            h_l_[type][k]->Fill((photo_->at(i)).lambda());
          } else {
            if (photon_[i].z() > 0) {
              nlong++;
            } else {
              nshort++;
            }
            h_x_[type][k]->Fill(photon_[i].x());
            h_y_[type][k]->Fill(photon_[i].y());
            h_z_[type][k]->Fill(photon_[i].z());
            h_t_[type][k]->Fill(photon_[i].t());
            h_l_[type][k]->Fill(photon_[i].lambda());
          }
        }
        h_nl_[type][k]->Fill(nlong);
        h_ns_[type][k]->Fill(nshort);
      }
    }
  }
}

void HFShowerLibraryAnalyzer::getRecord(int type, int record) {
  int nrc = record - 1;
  photon_.clear();
  photo_->clear();
  if (type > 0) {
    if (newForm_) {
      if (!v3version_) {
        hadBranch_->SetAddress(&photo_);
        hadBranch_->GetEntry(nrc + totEvents_);
      } else {
        std::vector<float> t;
        std::vector<float>* tp = &t;
        hadBranch_->SetAddress(&tp);
        hadBranch_->GetEntry(nrc + totEvents_);
        unsigned int tSize = t.size() / 5;
        photo_->reserve(tSize);
        for (unsigned int i = 0; i < tSize; i++) {
          photo_->push_back(
              HFShowerPhoton(t[i], t[1 * tSize + i], t[2 * tSize + i], t[3 * tSize + i], t[4 * tSize + i]));
        }
      }
    } else {
      hadBranch_->SetAddress(&photon_);
      hadBranch_->GetEntry(nrc);
    }
  } else {
    if (newForm_) {
      if (!v3version_) {
        emBranch_->SetAddress(&photo_);
        emBranch_->GetEntry(nrc);
      } else {
        std::vector<float> t;
        std::vector<float>* tp = &t;
        emBranch_->SetAddress(&tp);
        emBranch_->GetEntry(nrc);
        unsigned int tSize = t.size() / 5;
        photo_->reserve(tSize);
        for (unsigned int i = 0; i < tSize; i++) {
          photo_->push_back(
              HFShowerPhoton(t[i], t[1 * tSize + i], t[2 * tSize + i], t[3 * tSize + i], t[4 * tSize + i]));
        }
      }
    } else {
      emBranch_->SetAddress(&photon_);
      emBranch_->GetEntry(nrc);
    }
  }
  if (verbose_) {
    int nPhoton = (newForm_) ? photo_->size() : photon_.size();
    edm::LogVerbatim("HFShower") << "getRecord: Record " << record << " of type " << type << " with " << nPhoton
                                 << " photons";
    for (int j = 0; j < nPhoton; j++)
      if (newForm_)
        edm::LogVerbatim("HFShower") << "Photon " << j << " " << photo_->at(j);
      else
        edm::LogVerbatim("HFShower") << "Photon " << j << " " << photon_[j];
  }
}

void HFShowerLibraryAnalyzer::loadEventInfo(TBranch* branch) {
  if (branch) {
    std::vector<HFShowerLibraryEventInfo> eventInfoCollection;
    branch->SetAddress(&eventInfoCollection);
    branch->GetEntry(0);
    edm::LogVerbatim("HFShower") << "HFShowerLibrary::loadEventInfo loads EventInfo Collection of size "
                                 << eventInfoCollection.size() << " records";
    totEvents_ = eventInfoCollection[0].totalEvents();
    nMomBin_ = eventInfoCollection[0].numberOfBins();
    evtPerBin_ = eventInfoCollection[0].eventsPerBin();
    libVers_ = eventInfoCollection[0].showerLibraryVersion();
    listVersion_ = eventInfoCollection[0].physListVersion();
    pmom_ = eventInfoCollection[0].energyBins();
  } else {
    edm::LogVerbatim("HFShower") << "HFShowerLibrary::loadEventInfo loads EventInfo from hardwired numbers";
    nMomBin_ = 16;
    totEvents_ = nMomBin_ * evtPerBin_;
    libVers_ = 1.1;
    listVersion_ = 3.6;
    pmom_ = {2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 250, 350, 500, 1000};
  }
  for (int i = 0; i < nMomBin_; i++)
    pmom_[i] *= CLHEP::GeV;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFShowerLibraryAnalyzer);
