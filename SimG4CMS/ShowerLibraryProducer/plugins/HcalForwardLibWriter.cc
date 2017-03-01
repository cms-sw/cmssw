#include "SimG4CMS/ShowerLibraryProducer/interface/HcalForwardLibWriter.h"

#include "TFile.h"
#include "TTree.h"

HcalForwardLibWriter::HcalForwardLibWriter(const edm::ParameterSet& iConfig) {
  edm::ParameterSet theParms = iConfig.getParameter<edm::ParameterSet> ("HcalForwardLibWriterParameters");
  edm::FileInPath fp         = theParms.getParameter<edm::FileInPath> ("FileName");
  nbins                      = theParms.getParameter<int>("Nbins");
  nshowers                   = theParms.getParameter<int>("Nshowers");

  std::string pName = fp.fullPath();
  if (pName.find(".") == 0)
    pName.erase(0, 2);
  theDataFile = pName;
  readUserData();

  int bsize = 64000;
  fs->file().cd();
  LibTree = new TTree("HFSimHits", "HFSimHits");
  LibTree->Branch("emParticles", "HFShowerPhotons-emParticles", &emColl, bsize);
  LibTree->Branch("hadParticles", "HFShowerPhotons-hadParticles", &hadColl, bsize);
}

HcalForwardLibWriter::~HcalForwardLibWriter() {
}

void HcalForwardLibWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
// Event info
  std::vector<double> en;
  double momBin[16] = {2,3,5,7,10,15,20,30,50,75,100,150,250,350,500,1000};
  for (int i = 0; i < nbins; ++i) en.push_back(momBin[i]);

  //shower photons
  int n = theFileHandle.size();
// cycle over files ++++++++++++++++++++++++++++++++++++++++++++++++++++
  for (int i = 0; i < n; ++i) {
    std::string fn = theFileHandle[i].name;
    std::string particle = theFileHandle[i].id;
    TFile* theFile = new TFile(fn.c_str(), "READ");
    TTree *theTree = (TTree*)gDirectory->Get("g4SimHits/CherenkovPhotons");
    int nphot = 0;
    float x[10000];
    float y[10000];
    float z[10000];
    float t[10000];
    float lambda[10000];
    int fiberId[10000];
    for (int kk = 0; kk < 10000; ++kk) {
      x[kk] = 0.;
      y[kk] = 0.;
      z[kk] = 0.;
      t[kk] = 0.;
      lambda[kk] = 0.;
      fiberId[kk] = 0;
    }
    theTree->SetBranchAddress("nphot", &nphot);
    theTree->SetBranchAddress("x", &x);
    theTree->SetBranchAddress("y", &y);
    theTree->SetBranchAddress("z", &z);
    theTree->SetBranchAddress("t", &t);
    theTree->SetBranchAddress("lambda", &lambda);
    theTree->SetBranchAddress("fiberId", &fiberId);
    int nentries = int(theTree->GetEntries());
    if(nentries>5000) nentries=5000;
    int nbytes = 0;
// cycle over showers ====================================================
      for (int iev = 0; iev < nentries; iev++) {
        nbytes += theTree->GetEntry(iev);
	if (particle == "electron") {
	  emColl.clear();
	} else {
	  hadColl.clear();
        }
	float nphot_long=0;
	float nphot_short=0;
// cycle over photons in shower -------------------------------------------
	for (int iph = 0; iph < nphot; ++iph) {

	  if (fiberId[iph]==1) {
	    nphot_long++;
	  } else {
	    nphot_short++;
	    z[iph]= -z[iph];
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
	  LibTree->SetBranchStatus("hadParticles",0);
	} else {
	  LibTree->SetBranchStatus("emParticles",0);
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

void HcalForwardLibWriter::beginJob() {
}

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
