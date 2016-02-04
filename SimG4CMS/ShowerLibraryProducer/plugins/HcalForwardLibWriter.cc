#include "SimG4CMS/ShowerLibraryProducer/interface/HcalForwardLibWriter.h"

#include "TFile.h"
#include "TTree.h"

HcalForwardLibWriter::HcalForwardLibWriter(const edm::ParameterSet& iConfig) {
  edm::ParameterSet theParms = iConfig.getParameter<edm::ParameterSet> ("HcalForwardLibWriterParameters");
  edm::FileInPath fp = theParms.getParameter<edm::FileInPath> ("FileName");
  std::string pName = fp.fullPath();
  if (pName.find(".") == 0)
    pName.erase(0, 2);
  theDataFile = pName;
  readUserData();
  produces<HFShowerPhotonCollection> ("emParticles");
  produces<HFShowerPhotonCollection> ("hadParticles");
  produces<HFShowerLibraryEventInfo> ("EventInfo");

}

HcalForwardLibWriter::~HcalForwardLibWriter() {

}

void HcalForwardLibWriter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::auto_ptr<HFShowerPhotonCollection> product_em(new HFShowerPhotonCollection);
  std::auto_ptr<HFShowerPhotonCollection> product_had(new HFShowerPhotonCollection);
  
  //EventInfo
  std::auto_ptr<HFShowerLibraryEventInfo> product_evtInfo(new HFShowerLibraryEventInfo);
  float hfShowerLibV = 1.1;
  float phyListV = 3.6;
  std::vector<double> en;
  double momBin[12] = { 10., 15., 20., 35., 50., 80., 100., 150., 250., 350., 500., 1000. };
  for (int i = 0; i < 12; ++i)
    en.push_back(momBin[i]);
  HFShowerLibraryEventInfo evtInfo(60000, 12, 5000, hfShowerLibV, phyListV, en);
  *product_evtInfo = evtInfo;
  iEvent.put(product_evtInfo, "EventInfo");


  //shower photons
  HFShowerPhotonCollection emColl;
  HFShowerPhotonCollection hadColl;

  int n = theFileHandle.size();
  for (int i = 0; i < n; ++i) {
    std::string fn = theFileHandle[i].name;
    std::string particle = theFileHandle[i].id;
    //    int momBin = theFileHandle[i].momentum;
    TFile* theFile = new TFile(fn.c_str(), "READ");
    TTree* theTree = (TTree*) theFile->FindObjectAny("CherenkovPhotons");
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
    int nentry = int(theTree->GetEntries());
    if (particle == "electron") {
      for (int iev = 0; iev < nentry; iev++) {
	for (int iph = 0; iph < nphot; ++iph) {
	  HFShowerPhoton::Point pos(x[iph], y[iph], z[iph]);
	  HFShowerPhoton aPhoton(pos, t[iph], lambda[iph]);
	  emColl.push_back(aPhoton);
	}
      }
    }
    if (particle == "pion") {
    }
    theFile->Close();
    if (theFile)
      delete theFile;
    if (theTree)
      delete theTree;
  }
  *product_em = emColl;
  *product_had = hadColl;
  
  //fillEvent(product_em,product_had);
  //fillEvent(product_em, product_had);
  iEvent.put(product_em, "emParticles");
  iEvent.put(product_had, "hadParticles");
}

void HcalForwardLibWriter::beginJob() {
}
//void HcalForwardLibWriter::fillEvent(HFShowerPhotonCollection& em, HFShowerPhotonCollection& had) {
//
//	HFShowerPhotonCollection emColl;
//	HFShowerPhotonCollection hadColl;
//
//	int n = theFileHandle.size();
//	for (int i = 0; i < n; ++i) {
//		std::string fn = theFileHandle[i].name;
//		std::string particle = theFileHandle[i].id;
//		int momBin = theFileHandle[i].momentum;
//		TFile* theFile = new TFile(fn.c_str(), "READ");
//		TTree* theTree = (TTree*) theFile->FindObjectAny("CherenkovPhotons");
//		int nphot = 0;
//		float x[10000];
//		float y[10000];
//		float z[10000];
//		float t[10000];
//		float lambda[10000];
//		int fiberId[10000];
//		for (int kk = 0; kk < 10000; ++kk) {
//			x[kk] = 0.;
//			y[kk] = 0.;
//			z[kk] = 0.;
//			t[kk] = 0.;
//			lambda[kk] = 0.;
//			fiberId[kk] = 0;
//		}
//		theTree->SetBranchAddress("nphot", &nphot);
//		theTree->SetBranchAddress("x", &x);
//		theTree->SetBranchAddress("y", &y);
//		theTree->SetBranchAddress("z", &z);
//		theTree->SetBranchAddress("t", &t);
//		theTree->SetBranchAddress("lambda", &lambda);
//		theTree->SetBranchAddress("fiberId", &fiberId);
//		int nentry = int(theTree->GetEntries());
//		if (particle == "electron") {
//			for (int iev = 0; iev < nentry; iev++) {
//				for (int iph = 0; iph < nphot; ++iph) {
//					HFShowerPhoton::Point pos(x[iph], y[iph], z[iph]);
//					HFShowerPhoton aPhoton(pos, t[iph], lambda[iph]);
//					emColl.push_back(aPhoton);
//				}
//			}
//		}
//		if (particle == "pion") {
//		}
//		theFile->Close();
//		if (theFile)
//			delete theFile;
//		if (theTree)
//			delete theTree;
//	}
//	*em = emColl;
//	*had = hadColl;
//}

void HcalForwardLibWriter::endJob() {
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
