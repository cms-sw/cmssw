#include "SimG4CMS/ShowerLibraryProducer/interface/HcalForwardLibWriter.h"

#include "TFile.h"
#include "TTree.h"

HcalForwardLibWriter::HcalForwardLibWriter(const edm::ParameterSet& iConfig) {
  edm::ParameterSet theParms = iConfig.getParameter<edm::ParameterSet>("HcalForwardLibWriterParameters");
  edm::FileInPath fp = theParms.getParameter<edm::FileInPath>("FileName");
  nbins = theParms.getParameter<int>("Nbins");
  nshowers = theParms.getParameter<int>("Nshowers");
  bsize = theParms.getParameter<int>("BufSize");
  splitlevel = theParms.getParameter<int>("SplitLevel");

  std::string pName = fp.fullPath();
  if (pName.find('.') == 0)
    pName.erase(0, 2);
  theDataFile = pName;
  readUserData();

  fs->file().cd();
  LibTree = new TTree("HFSimHits", "HFSimHits");

  //https://root.cern/root/html534/TTree.html
  // TBranch*Branch(const char* name, const char* classname, void** obj, Int_t bufsize = 32000, Int_t splitlevel = 99)
  LibTree->Branch("emParticles", "HFShowerPhotons-emParticles", &emColl, bsize, splitlevel);
  LibTree->Branch("hadParticles", "HFShowerPhotons-hadParticles", &hadColl, bsize, splitlevel);
}

HcalForwardLibWriter::~HcalForwardLibWriter() {}

void HcalForwardLibWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Event info
  std::vector<double> en;
  double momBin[16] = {2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 250, 350, 500, 1000};
  en.reserve(nbins);
  for (int i = 0; i < nbins; ++i)
    en.push_back(momBin[i]);

  //shower photons
  int n = theFileHandle.size();
  // cycle over files ++++++++++++++++++++++++++++++++++++++++++++++++++++
  for (int i = 0; i < n; ++i) {
    std::string fn = theFileHandle[i].name;
    std::string particle = theFileHandle[i].id;

    //    std::cout << "*** Input file  " << i << "   " << fn << std::endl;

    TFile* theFile = new TFile(fn.c_str(), "READ");
    TTree* theTree = (TTree*)gDirectory->Get("g4SimHits/CherenkovPhotons");
    int nphot = 0;

    const int size = 10000;
    if (nshowers > size) {
      edm::LogError("HcalForwardLibWriter") << "Too big Nshowers number";
      return;
    }

    float x[size] = {0.};
    float y[size] = {0.};
    float z[size] = {0.};
    float t[size] = {0.};
    float lambda[size] = {0.};
    int fiberId[size] = {0};
    float primZ;  // added

    theTree->SetBranchAddress("nphot", &nphot);
    theTree->SetBranchAddress("x", &x);
    theTree->SetBranchAddress("y", &y);
    theTree->SetBranchAddress("z", &z);
    theTree->SetBranchAddress("t", &t);
    theTree->SetBranchAddress("lambda", &lambda);
    theTree->SetBranchAddress("fiberId", &fiberId);
    theTree->SetBranchAddress("primZ", &primZ);  // added
    int nentries = int(theTree->GetEntries());
    int ngood = 0;
    int nbytes = 0;
    // cycle over showers ====================================================
    for (int iev = 0; iev < nentries; iev++) {
      nbytes += theTree->GetEntry(iev);
      if (primZ < 990.)
        continue;  // exclude showers with interactions in front of HF (1m of air)
      ngood++;
      if (ngood > nshowers)
        continue;
      if (particle == "electron") {
        emColl.clear();
      } else {
        hadColl.clear();
      }
      float nphot_long = 0;
      float nphot_short = 0;
      // cycle over photons in shower -------------------------------------------
      for (int iph = 0; iph < nphot; ++iph) {
        if (fiberId[iph] == 1) {
          nphot_long++;
        } else {
          nphot_short++;
          z[iph] = -z[iph];
        }

        HFShowerPhoton::Point pos(x[iph], y[iph], z[iph]);
        HFShowerPhoton aPhoton(pos, t[iph], lambda[iph]);
        if (particle == "electron") {
          emColl.push_back(aPhoton);
        } else {
          hadColl.push_back(aPhoton);
        }
      }
      // end of cycle over photons in shower -------------------------------------------

      //        if(iev>0) LibTree->SetBranchStatus("HFShowerLibInfo",0);
      if (particle == "electron") {
        LibTree->SetBranchStatus("hadParticles", false);
      } else {
        LibTree->SetBranchStatus("emParticles", false);
      }
      LibTree->Fill();
      if (particle == "electron") {
        emColl.clear();
      } else {
        hadColl.clear();
      }
    }
    // end of cycle over showers ====================================================
    theFile->Close();
  }
  // end of cycle over files ++++++++++++++++++++++++++++++++++++++++++++++++++++
}

void HcalForwardLibWriter::beginJob() {}

void HcalForwardLibWriter::endJob() {
  fs->file().cd();
  LibTree->Write();
  LibTree->Print();
}

int HcalForwardLibWriter::readUserData(void) {
  std::ifstream input(theDataFile.c_str());
  if (input.fail()) {
    return 0;
  }
  std::string theFileName, thePID;
  int mom;
  int k = 0;
  while (!input.eof()) {
    input >> theFileName >> thePID >> mom;
    if (!input.fail()) {
      FileHandle aFile;
      aFile.name = theFileName;
      aFile.id = thePID;
      aFile.momentum = mom;
      theFileHandle.push_back(aFile);
      ++k;
    } else {
      input.clear();
    }
    input.ignore(999, '\n');
  }
  return k;
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalForwardLibWriter);
