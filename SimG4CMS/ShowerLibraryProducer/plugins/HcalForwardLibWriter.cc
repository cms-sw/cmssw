// -*- C++ -*-
//
// Package:    HcalForwardLibWriter
// Class:      HcalForwardLibWriter
// 
/**\class HcalForwardLibWriter HcalForwardLibWriter.cc SimG4CMS/ShowerLibraryProducer/plugins/HcalForwardLibWriter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Taylan Yetkin,510 1-004,+41227672815,
//         Created:  Thu Feb  9 13:02:38 CET 2012
// $Id: HcalForwardLibWriter.cc,v 1.5 2013/05/25 17:03:42 chrjones Exp $
//
//

#include "SimG4CMS/ShowerLibraryProducer/interface/HcalForwardLibWriter.h"

HcalForwardLibWriter::HcalForwardLibWriter(const edm::ParameterSet& iConfig) {

  edm::ParameterSet theParms = iConfig.getParameter<edm::ParameterSet> ("HcalForwardLibWriterParameters");
  edm::FileInPath fp = theParms.getParameter<edm::FileInPath> ("FileName");
  std::string pName = fp.fullPath();
  std::cout<<pName<<std::endl;
  fDataFile = pName;
  readUserData();
    
  //register shower library products with label
  produces<HFShowerPhotonCollection> ("emParticles");
  produces<std::vector<HFShowerPhotonCollection> > ("emParticlesss");
  produces< std::vector<int> > ("emParticles");
  produces<HFShowerPhotonCollection> ("hadParticles");
  produces< std::vector<int> > ("hadParticles");
  produces<std::vector<HFShowerLibraryEventInfo> > ("HFShowerLibraryEventInfo");
}

HcalForwardLibWriter::~HcalForwardLibWriter() {}

void HcalForwardLibWriter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  std::vector<double> energyBin;
  double energyBin2[12] = {10,15,20,35,50,80,100,150,250,350,500,1000};
  for (int z = 0; z< 12; ++z) {energyBin.push_back(energyBin2[z]);}

  std::vector<HFShowerLibraryEventInfo> Info;
  HFShowerPhotonCollection emColl;
  std::vector<HFShowerPhotonCollection> emCollsss;
  HFShowerPhotonCollection fakeColl;
  HFShowerPhoton afakePhoton(1,3,5,8,480);
  fakeColl.push_back(afakePhoton);
  HFShowerPhotonCollection hadColl;
  //HFShowerPhotonCollection emCollnphot;
  //std::vector<HFShowerPhotonCollection> emCollnphot;
  //std::vector<int> emCollnPhoton;
  std::vector<int> emCollnPhoton;
  std::vector<int> hadCollnPhoton;

  int n = fFileHandle.size();
  for (int i = 0; i < n; ++i) {
    std::string fn = fFileHandle[i].name;
    std::cout<<fn<<std::endl;
    std::string particle = fFileHandle[i].id;
    /*
    int momBin = fFileHandle[i].momentum;
    energyBin.push_back(momBin);
    */
    fFile = new TFile(fn.c_str(), "READ");
    fTree = (TTree*) fFile->FindObjectAny("CherenkovPhotons");
    if(!fTree){
      throw cms::Exception("NullPointer") 
	<< "Cannot find TTree with name CherenkovPhotons";
    }
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
    fTree->SetBranchAddress("nphot", &nphot);
    fTree->SetBranchAddress("x", &x);
    fTree->SetBranchAddress("y", &y);
    fTree->SetBranchAddress("z", &z);
    fTree->SetBranchAddress("t", &t);
    fTree->SetBranchAddress("lambda", &lambda);
    fTree->SetBranchAddress("fiberId", &fiberId);
    int nentry = int(fTree->GetEntries());
    std::cout<<"nenetry   " << nentry<<std::endl;
    if (particle == "electron") {
      for (int iev = 0; iev < nentry; iev++) {
	fTree->GetEntry(iev);
	std::cout<<"nphot  "<<nphot<<std::endl;
	emCollnPhoton.push_back(nphot);
	for (int iph = 0; iph < nphot; ++iph) {
	  HFShowerPhoton::Point pos(x[iph], y[iph], z[iph]);
	  HFShowerPhoton aPhoton(pos, t[iph], lambda[iph]);
	  emColl.push_back(aPhoton);
	}
	emCollsss.push_back(emColl);
	emColl.clear();
      }
    }
    if (particle == "pion") {
      for (int iev = 0; iev < nentry; iev++) {
	fTree->GetEntry(iev);
	hadCollnPhoton.push_back(nphot);
	for (int iph = 0; iph < nphot; ++iph) {
	  HFShowerPhoton::Point pos(x[iph], y[iph], z[iph]);
	  HFShowerPhoton aPhoton(pos, t[iph], lambda[iph]);
	  hadColl.push_back(aPhoton);
	}
      }
    }
    
  }

  HFShowerLibraryEventInfo aInfo((n/2)*5000,n/2,5000,1,1,energyBin);
  Info.push_back(aInfo);

  std::auto_ptr< std::vector<HFShowerLibraryEventInfo> > product_info(new std::vector<HFShowerLibraryEventInfo>(Info) );
  std::auto_ptr<HFShowerPhotonCollection > product_em(new HFShowerPhotonCollection(fakeColl));
  std::auto_ptr<std::vector<HFShowerPhotonCollection> > product_emsss(new std::vector<HFShowerPhotonCollection>(emCollsss));
  std::cout<<"em coll size "<<emCollsss.size()<<std::endl;
  //std::auto_ptr< std::vector<int> > product_em_nphot(new std::vector<int>(emCollnPhoton));
  std::auto_ptr<HFShowerPhotonCollection> product_had(new HFShowerPhotonCollection(hadColl));
  //std::auto_ptr<std::vector<int> > product_had_nphot(new std::vector<int>(hadCollnPhoton));
  iEvent.put(product_info, "HFShowerLibraryEventInfo");   
  iEvent.put(product_emsss, "emParticles");
  iEvent.put(product_had, "hadParticles");
  //iEvent.put(product_em_nphot, "emParticles");
  //iEvent.put(product_had_nphot, "hadParticles");

}

void HcalForwardLibWriter::readUserData() {
  std::cout << " using " <<std::endl;
  std::ifstream input(fDataFile.c_str());
  if (input.fail()) {
    throw cms::Exception("MissingFile")
      << "Cannot find file" << fDataFile.c_str();
  }
  std::string fFileName, fPID;
  int fMom;
  while (!input.eof()) {
    input >> fFileName >> fPID >> fMom;
    std::cout << " using " <<  fFileName << " fPID" << fPID << " fMom" << fMom  << std::endl;
    if (!input.fail()) {
      std::cout << " using " <<  fFileName << " " << fPID << " " << fMom  << std::endl;
      FileHandle aFile;
      aFile.name = fFileName;
      aFile.id = fPID;
      aFile.momentum = fMom;
      fFileHandle.push_back(aFile);
    } else {
      input.clear();
    }
    input.ignore(999, '\n');
  }
}

void HcalForwardLibWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalForwardLibWriter);
