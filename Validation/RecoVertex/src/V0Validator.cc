// -*- C++ -*-
//
// Package:    V0Validator
// Class:      V0Validator
// 
/**\class V0Validator V0Validator.cc Validation/RecoVertex/src/V0Validator.cc

 Description: Creates validation histograms for RecoVertex/V0Producer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Wed Feb 18 17:21:04 MST 2009
// $Id: V0Validator.cc,v 1.9.4.1 2013/05/14 15:26:38 speer Exp $
//
//


#include "Validation/RecoVertex/interface/V0Validator.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

typedef std::vector<TrackingVertex> TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection> TrackingVertexRef;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenParticle > GenParticleRefVector;

const double piMass = 0.13957018;
const double piMassSquared = piMass*piMass;
const double protonMass = 0.93827203;
const double protonMassSquared = protonMass*protonMass;



V0Validator::V0Validator(const edm::ParameterSet& iConfig) : 
  theDQMRootFileName(iConfig.getParameter<std::string>("DQMRootFileName")),
  k0sCollectionTag(iConfig.getParameter<edm::InputTag>("kShortCollection")),
  lamCollectionTag(iConfig.getParameter<edm::InputTag>("lambdaCollection")),
  dirName(iConfig.getParameter<std::string>("dirName")) {
  genLam = genK0s = realLamFoundEff = realK0sFoundEff = lamCandFound = 
    k0sCandFound = noTPforK0sCand = noTPforLamCand = realK0sFound = realLamFound = 0;
  theDQMstore = edm::Service<DQMStore>().operator->();
}


V0Validator::~V0Validator() {

}

//void V0Validator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
//void V0Validator::beginJob(const edm::EventSetup& iSetup) {
//}

//void V0Validator::beginJob(const edm::EventSetup& iSetup) {
void V0Validator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  //std::cout << "Running V0Validator" << std::endl;
  //theDQMstore = edm::Service<DQMStore>().operator->();
  //std::cout << "In beginJob() at line 1" << std::endl;
  //edm::Service<TFileService> fs;

  theDQMstore->cd();
  std::string subDirName = dirName + "/EffFakes";
  theDQMstore->setCurrentFolder(subDirName.c_str());

  ksEffVsR = theDQMstore->book1D("K0sEffVsR", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEta = theDQMstore->book1D("K0sEffVsEta",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPt = theDQMstore->book1D("K0sEffVsPt",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;

  ksTkEffVsR = theDQMstore->book1D("K0sTkEffVsR", 
			  "K^{0}_{S} Tracking Efficiency vs #rho", 40, 0., 40.);
  ksTkEffVsEta = theDQMstore->book1D("K0sTkEffVsEta",
			    "K^{0}_{S} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  ksTkEffVsPt = theDQMstore->book1D("K0sTkEffVsPt",
			   "K^{0}_{S} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  ksEffVsR_num = theDQMstore->book1D("K0sEffVsR_num", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEta_num = theDQMstore->book1D("K0sEffVsEta_num",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPt_num = theDQMstore->book1D("K0sEffVsPt_num",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;

  ksTkEffVsR_num = theDQMstore->book1D("K0sTkEffVsR_num", 
			  "K^{0}_{S} Tracking Efficiency vs #rho", 40, 0., 40.);
  ksTkEffVsEta_num = theDQMstore->book1D("K0sTkEffVsEta_num",
			    "K^{0}_{S} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  ksTkEffVsPt_num = theDQMstore->book1D("K0sTkEffVsPt_num",
			   "K^{0}_{S} Tracking Efficiency vs p_{T}", 70, 0., 20.);;


  ksEffVsR_denom = theDQMstore->book1D("K0sEffVsR_denom", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEta_denom = theDQMstore->book1D("K0sEffVsEta_denom",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPt_denom = theDQMstore->book1D("K0sEffVsPt_denom",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;


  lamEffVsR = theDQMstore->book1D("LamEffVsR",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEta = theDQMstore->book1D("LamEffVsEta",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPt = theDQMstore->book1D("LamEffVsPt",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);


  lamTkEffVsR = theDQMstore->book1D("LamTkEffVsR",
			   "#Lambda^{0} TrackingEfficiency vs #rho", 40, 0., 40.);
  lamTkEffVsEta = theDQMstore->book1D("LamTkEffVsEta",
			     "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  lamTkEffVsPt = theDQMstore->book1D("LamTkEffVsPt",
			    "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  lamEffVsR_num = theDQMstore->book1D("LamEffVsR_num",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEta_num = theDQMstore->book1D("LamEffVsEta_num",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPt_num = theDQMstore->book1D("LamEffVsPt_num",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);


  lamTkEffVsR_num = theDQMstore->book1D("LamTkEffVsR_num",
			   "#Lambda^{0} TrackingEfficiency vs #rho", 40, 0., 40.);
  lamTkEffVsEta_num = theDQMstore->book1D("LamTkEffVsEta_num",
			     "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  lamTkEffVsPt_num = theDQMstore->book1D("LamTkEffVsPt_num",
			    "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);


  lamEffVsR_denom = theDQMstore->book1D("LamEffVsR_denom",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEta_denom = theDQMstore->book1D("LamEffVsEta_denom",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPt_denom = theDQMstore->book1D("LamEffVsPt_denom",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);

  //theDQMstore->cd();
  //subDirName = dirName + "/Fake";
  //theDQMstore->setCurrentFolder(subDirName.c_str());


  ksFakeVsR = theDQMstore->book1D("K0sFakeVsR",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEta = theDQMstore->book1D("K0sFakeVsEta",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPt = theDQMstore->book1D("K0sFakeVsPt",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);
  ksTkFakeVsR = theDQMstore->book1D("K0sTkFakeVsR",
			   "K^{0}_{S} Tracking Fake Rate vs #rho", 40, 0., 40.);
  ksTkFakeVsEta = theDQMstore->book1D("K0sTkFakeVsEta",
			     "K^{0}_{S} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  ksTkFakeVsPt = theDQMstore->book1D("K0sTkFakeVsPt",
			    "K^{0}_{S} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  ksFakeVsR_num = theDQMstore->book1D("K0sFakeVsR_num",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEta_num = theDQMstore->book1D("K0sFakeVsEta_num",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPt_num = theDQMstore->book1D("K0sFakeVsPt_num",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);
  ksTkFakeVsR_num = theDQMstore->book1D("K0sTkFakeVsR_num",
			   "K^{0}_{S} Tracking Fake Rate vs #rho", 40, 0., 40.);
  ksTkFakeVsEta_num = theDQMstore->book1D("K0sTkFakeVsEta_num",
			     "K^{0}_{S} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  ksTkFakeVsPt_num = theDQMstore->book1D("K0sTkFakeVsPt_num",
			    "K^{0}_{S} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  ksFakeVsR_denom = theDQMstore->book1D("K0sFakeVsR_denom",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEta_denom = theDQMstore->book1D("K0sFakeVsEta_denom",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPt_denom = theDQMstore->book1D("K0sFakeVsPt_denom",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);

  lamFakeVsR = theDQMstore->book1D("LamFakeVsR",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEta = theDQMstore->book1D("LamFakeVsEta",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPt = theDQMstore->book1D("LamFakeVsPt",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);
  lamTkFakeVsR = theDQMstore->book1D("LamTkFakeVsR",
			    "#Lambda^{0} Tracking Fake Rate vs #rho", 40, 0., 40.);
  lamTkFakeVsEta = theDQMstore->book1D("LamTkFakeVsEta",
			      "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  lamTkFakeVsPt = theDQMstore->book1D("LamTkFakeVsPt",
			     "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  lamFakeVsR_num = theDQMstore->book1D("LamFakeVsR_num",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEta_num = theDQMstore->book1D("LamFakeVsEta_num",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPt_num = theDQMstore->book1D("LamFakeVsPt_num",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);
  lamTkFakeVsR_num = theDQMstore->book1D("LamTkFakeVsR_num",
			    "#Lambda^{0} Tracking Fake Rate vs #rho", 40, 0., 40.);
  lamTkFakeVsEta_num = theDQMstore->book1D("LamTkFakeVsEta_num",
			      "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  lamTkFakeVsPt_num = theDQMstore->book1D("LamTkFakeVsPt_num",
			     "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  lamFakeVsR_denom = theDQMstore->book1D("LamFakeVsR_denom",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEta_denom = theDQMstore->book1D("LamFakeVsEta_denom",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPt_denom = theDQMstore->book1D("LamFakeVsPt_denom",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  theDQMstore->cd();
  subDirName = dirName + "/Other";
  theDQMstore->setCurrentFolder(subDirName.c_str());

  nKs = theDQMstore->book1D("nK0s",
		     "Number of K^{0}_{S} found per event", 60, 0., 60.);
  nLam = theDQMstore->book1D("nLam",
		      "Number of #Lambda^{0} found per event", 60, 0., 60.);

  ksXResolution = theDQMstore->book1D("ksXResolution",
			       "Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  ksYResolution = theDQMstore->book1D("ksYResolution",
			       "Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  ksZResolution = theDQMstore->book1D("ksZResolution",
			       "Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  lamXResolution = theDQMstore->book1D("lamXResolution",
				"Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  lamYResolution = theDQMstore->book1D("lamYResolution",
				"Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  lamZResolution = theDQMstore->book1D("lamZResolution",
				"Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  ksAbsoluteDistResolution = theDQMstore->book1D("ksRResolution",
					  "Resolution of absolute distance from primary vertex to V0 vertex",
					  100, 0., 50.);
  lamAbsoluteDistResolution = theDQMstore->book1D("lamRResolution",
					   "Resolution of absolute distance from primary vertex to V0 vertex",
					   100, 0., 50.);

  ksCandStatus = theDQMstore->book1D("ksCandStatus",
			  "Fake type by cand status",
			  10, 0., 10.);
  lamCandStatus = theDQMstore->book1D("ksCandStatus",
			  "Fake type by cand status",
			  10, 0., 10.);

  double minKsMass = 0.49767 - 0.07;
  double maxKsMass = 0.49767 + 0.07;
  double minLamMass = 1.1156 - 0.05;
  double maxLamMass = 1.1156 + 0.05;
  int ksMassNbins = 100;
  double ksMassXmin = minKsMass;
  double ksMassXmax = maxKsMass;
  int lamMassNbins = 100;
  double lamMassXmin = minLamMass;
  double lamMassXmax = maxLamMass;

  fakeKsMass = theDQMstore->book1D("ksMassFake",
			     "Mass of fake K0S",
			     ksMassNbins, minKsMass, maxKsMass);
  goodKsMass = theDQMstore->book1D("ksMassGood",
			     "Mass of good reco K0S",
			     ksMassNbins, minKsMass, maxKsMass);
  fakeLamMass = theDQMstore->book1D("lamMassFake",
			      "Mass of fake Lambda",
			      lamMassNbins, minLamMass, maxLamMass);
  goodLamMass = theDQMstore->book1D("lamMassGood",
			      "Mass of good Lambda",
			      lamMassNbins, minLamMass, maxLamMass);

  ksMassAll = theDQMstore->book1D("ksMassAll",
				  "Invariant mass of all K0S",
				  ksMassNbins, ksMassXmin, ksMassXmax);
  lamMassAll = theDQMstore->book1D("lamMassAll",
				   "Invariant mass of all #Lambda^{0}",
				   lamMassNbins, lamMassXmin, lamMassXmax);

  ksFakeDauRadDist = theDQMstore->book1D("radDistFakeKs",
				   "Production radius of daughter particle of Ks fake",
				   100, 0., 15.);
  lamFakeDauRadDist = theDQMstore->book1D("radDistFakeLam",
				    "Production radius of daughter particle of Lam fake",
				    100, 0., 15.);

  /*

  ksEffVsRHist = new TH1F("K0sEffVsR", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEtaHist = new TH1F("K0sEffVsEta",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPtHist = new TH1F("K0sEffVsPt",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;
  ksFakeVsRHist = new TH1F("K0sFakeVsR",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEtaHist = new TH1F("K0sFakeVsEta",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPtHist = new TH1F("K0sFakeVsPt",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);

  ksTkEffVsRHist = new TH1F("K0sTkEffVsR", 
			  "K^{0}_{S} Tracking Efficiency vs #rho", 40, 0., 40.);
  ksTkEffVsEtaHist = new TH1F("K0sTkEffVsEta",
			    "K^{0}_{S} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  ksTkEffVsPtHist = new TH1F("K0sTkEffVsPt",
			   "K^{0}_{S} Tracking Efficiency vs p_{T}", 70, 0., 20.);;
  ksTkFakeVsRHist = new TH1F("K0sTkFakeVsR",
			   "K^{0}_{S} Tracking Fake Rate vs #rho", 40, 0., 40.);
  ksTkFakeVsEtaHist = new TH1F("K0sTkFakeVsEta",
			     "K^{0}_{S} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  ksTkFakeVsPtHist = new TH1F("K0sTkFakeVsPt",
			    "K^{0}_{S} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  ksEffVsRHist_denom = new TH1F("K0sEffVsR_denom", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEtaHist_denom = new TH1F("K0sEffVsEta_denom",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPtHist_denom = new TH1F("K0sEffVsPt_denom",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;
  ksFakeVsRHist_denom = new TH1F("K0sFakeVsR_denom",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEtaHist_denom = new TH1F("K0sFakeVsEta_denom",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPtHist_denom = new TH1F("K0sFakeVsPt_denom",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);

  lamEffVsRHist = new TH1F("LamEffVsR",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEtaHist = new TH1F("LamEffVsEta",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPtHist = new TH1F("LamEffVsPt",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);
  lamFakeVsRHist = new TH1F("LamFakeVsR",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEtaHist = new TH1F("LamFakeVsEta",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPtHist = new TH1F("LamFakeVsPt",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  lamTkEffVsRHist = new TH1F("LamTkEffVsR",
			   "#Lambda^{0} TrackingEfficiency vs #rho", 40, 0., 40.);
  lamTkEffVsEtaHist = new TH1F("LamTkEffVsEta",
			     "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  lamTkEffVsPtHist = new TH1F("LamTkEffVsPt",
			    "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);
  lamTkFakeVsRHist = new TH1F("LamTkFakeVsR",
			    "#Lambda^{0} Tracking Fake Rate vs #rho", 40, 0., 40.);
  lamTkFakeVsEtaHist = new TH1F("LamTkFakeVsEta",
			      "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  lamTkFakeVsPtHist = new TH1F("LamTkFakeVsPt",
			     "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  lamEffVsRHist_denom = new TH1F("LamEffVsR_denom",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEtaHist_denom = new TH1F("LamEffVsEta_denom",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPtHist_denom = new TH1F("LamEffVsPt_denom",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);
  lamFakeVsRHist_denom = new TH1F("LamFakeVsR_denom",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEtaHist_denom = new TH1F("LamFakeVsEta_denom",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPtHist_denom = new TH1F("LamFakeVsPt_denom",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  nKsHist = new TH1F("nK0s",
		     "Number of K^{0}_{S} found per event", 60, 0., 60.);
  nLamHist = new TH1F("nLam",
		      "Number of #Lambda^{0} found per event", 60, 0., 60.);

  ksXResolutionHist = new TH1F("ksXResolution",
			       "Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  ksYResolutionHist = new TH1F("ksYResolution",
			       "Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  ksZResolutionHist = new TH1F("ksZResolution",
			       "Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  lamXResolutionHist = new TH1F("lamXResolution",
				"Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  lamYResolutionHist = new TH1F("lamYResolution",
				"Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  lamZResolutionHist = new TH1F("lamZResolution",
				"Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  ksAbsoluteDistResolutionHist = new TH1F("ksRResolution",
					  "Resolution of absolute distance from primary vertex to V0 vertex",
					  100, 0., 50.);
  lamAbsoluteDistResolutionHist = new TH1F("lamRResolution",
					   "Resolution of absolute distance from primary vertex to V0 vertex",
					   100, 0., 50.);

  ksCandStatusHist = new TH1F("ksCandStatus",
			  "Fake type by cand status",
			  10, 0., 10.);
  lamCandStatusHist = new TH1F("ksCandStatus",
			  "Fake type by cand status",
			  10, 0., 10.);

  double minKsMass = 0.49767 - 0.07;
  double maxKsMass = 0.49767 + 0.07;
  double minLamMass = 1.1156 - 0.05;
  double maxLamMass = 1.1156 + 0.05;
  fakeKsMassHisto = new TH1F("ksMassFake",
			     "Mass of fake K0s",
			     100, minKsMass, maxKsMass);
  goodKsMassHisto = new TH1F("ksMassGood",
			     "Mass of good reco K0s",
			     100, minKsMass, maxKsMass);
  fakeLamMassHisto = new TH1F("lamMassFake",
			      "Mass of fake Lambda",
			      100, minLamMass, maxLamMass);
  goodLamMassHisto = new TH1F("lamMassGood",
			      "Mass of good Lambda",
			      100, minLamMass, maxLamMass);

  ksFakeDauRadDistHisto = new TH1F("radDistFakeKs",
				   "Production radius of daughter particle of Ks fake",
				   100, 0., 15.);
  lamFakeDauRadDistHisto = new TH1F("radDistFakeLam",
				    "Production radius of daughter particle of Lam fake",
				    100, 0., 15.);*/


  //std::cout << "Histograms booked" << std::endl;

  /*ksEffVsRHist->Sumw2();
  ksEffVsEtaHist->Sumw2();
  ksEffVsPtHist->Sumw2();
  ksTkEffVsRHist->Sumw2();
  ksTkEffVsEtaHist->Sumw2();
  ksTkEffVsPtHist->Sumw2();
  ksFakeVsRHist->Sumw2();
  ksFakeVsEtaHist->Sumw2();
  ksFakeVsPtHist->Sumw2();
  ksTkFakeVsRHist->Sumw2();
  ksTkFakeVsEtaHist->Sumw2();
  ksTkFakeVsPtHist->Sumw2();

  lamEffVsRHist->Sumw2();
  lamEffVsEtaHist->Sumw2();
  lamEffVsPtHist->Sumw2();
  lamTkEffVsRHist->Sumw2();
  lamTkEffVsEtaHist->Sumw2();
  lamTkEffVsPtHist->Sumw2();
  lamFakeVsRHist->Sumw2();
  lamFakeVsEtaHist->Sumw2();
  lamFakeVsPtHist->Sumw2();
  lamTkFakeVsRHist->Sumw2();
  lamTkFakeVsEtaHist->Sumw2();
  lamTkFakeVsPtHist->Sumw2();*/

}

void V0Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using std::cout;
  using std::endl;
  using namespace edm;
  using namespace std;

  //cout << "In analyze(), getting collections..." << endl;
  // Get event setup info, B-field and tracker geometry
  ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

  // Make matching collections
  //reco::RecoToSimCollection recSimColl;
  //reco::SimToRecoCollection simRecColl;
   
  Handle<reco::RecoToSimCollection > recotosimCollectionH;
  iEvent.getByLabel("trackingParticleRecoTrackAsssociation", recotosimCollectionH);
  //recSimColl= *( recotosimCollectionH.product() ); 
  
  Handle<reco::SimToRecoCollection> simtorecoCollectionH;
  iEvent.getByLabel("trackingParticleRecoTrackAsssociation", simtorecoCollectionH);
  //simRecColl= *( simtorecoCollectionH.product() );

  edm::Handle<TrackingParticleCollection>  TPCollectionEff ;
  iEvent.getByLabel("mix", "MergedTrackTruth", TPCollectionEff);
  const TrackingParticleCollection tPCeff = *( TPCollectionEff.product() );

  edm::ESHandle<TrackAssociatorBase> associatorByHits;
  iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits", associatorByHits);

  //VertexAssociatorBase* associatorByTracks;

  //  edm::ESHandle<VertexAssociatorBase> theTracksAssociator;
  //  iSetup.get<VertexAssociatorRecord>().get("VertexAssociatorByTracks",theTracksAssociator);
  //  associatorByTracks = (VertexAssociatorBase *) theTracksAssociator.product();

  // Get tracks
  Handle< View<reco::Track> > trackCollectionH;
  iEvent.getByLabel("generalTracks", trackCollectionH);

  Handle<SimTrackContainer> simTrackCollection;
  iEvent.getByLabel("g4SimHits", simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());

  Handle<SimVertexContainer> simVertexCollection;
  iEvent.getByLabel("g4SimHits", simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());

  //Get tracking particles
  //  -->tracks
  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  iEvent.getByLabel("mix", "MergedTrackTruth", TPCollectionH);
  const View<reco::Track>  tC = *( trackCollectionH.product() );

//  edm::Handle<TrackingVertexCollection>  TVCollectionH ;
//  iEvent.getByLabel("trackingParticles","VertexTruth",TVCollectionH);
//  const TrackingVertexCollection tVC   = *(TVCollectionH.product());

  // Select the primary vertex, create a new reco::Vertex to hold it
  edm::Handle< std::vector<reco::Vertex> > primaryVtxCollectionH;
  iEvent.getByLabel("offlinePrimaryVertices", primaryVtxCollectionH);
  const reco::VertexCollection primaryVertexCollection   = *(primaryVtxCollectionH.product());

  reco::Vertex* thePrimary = 0;
  std::vector<reco::Vertex>::const_iterator iVtxPH = primaryVtxCollectionH->begin();
  for(std::vector<reco::Vertex>::const_iterator iVtx = primaryVtxCollectionH->begin();
      iVtx < primaryVtxCollectionH->end();
      iVtx++) {
    if(primaryVtxCollectionH->size() > 1) {
      if(iVtx->tracksSize() > iVtxPH->tracksSize()) {
	iVtxPH = iVtx;
      }
    }
    else iVtxPH = iVtx;
  }
  thePrimary = new reco::Vertex(*iVtxPH);

  //cout << "Done with collections, associating reco and sim..." << endl;
 

  //reco::RecoToSimCollection r2s = associatorByHits->associateRecoToSim(trackCollectionH,TPCollectionH,&iEvent );
  //reco::SimToRecoCollection s2r = associatorByHits->associateSimToReco(trackCollectionH,TPCollectionH,&iEvent );

//  reco::VertexRecoToSimCollection vr2s = associatorByTracks->associateRecoToSim(primaryVtxCollectionH, TVCollectionH, iEvent, r2s);
//  reco::VertexSimToRecoCollection vs2r = associatorByTracks->associateSimToReco(primaryVtxCollectionH, TVCollectionH, iEvent, s2r);

  //get the V0s;   
  edm::Handle<reco::VertexCompositeCandidateCollection> k0sCollection;
  edm::Handle<reco::VertexCompositeCandidateCollection> lambdaCollection;
  //iEvent.getByLabel("generalV0Candidates", "Kshort", k0sCollection);
  //iEvent.getByLabel("generalV0Candidates", "Lambda", lambdaCollection);
  iEvent.getByLabel(k0sCollectionTag, k0sCollection);
  iEvent.getByLabel(lamCollectionTag, lambdaCollection);

  //make vector of pair of trackingParticles to hold good V0 candidates
  std::vector< pair<TrackingParticleRef, TrackingParticleRef> > trueK0s;
  std::vector< pair<TrackingParticleRef, TrackingParticleRef> > trueLams;
  std::vector<double> trueKsMasses;
  std::vector<double> trueLamMasses;

  ////////////////////////////
  // Do vertex calculations //
  ////////////////////////////
/*
  if( k0sCollection->size() > 0 ) {
    for(reco::VertexCompositeCandidateCollection::const_iterator iK0s = k0sCollection->begin();
	iK0s != k0sCollection->end();
	iK0s++) {
      // Still can't actually associate the V0 vertex with a TrackingVertexCollection.
      //  Is this a problem?  You bet.
      reco::VertexCompositeCandidate::CovarianceMatrix aErr;
      iK0s->fillVertexCovariance(aErr);
      reco::Vertex tVtx(iK0s->vertex(), aErr);
      reco::VertexCollection *tVtxColl = 0;
      tVtxColl->push_back(tVtx);
      reco::VertexRef aVtx(tVtxColl, 0);
      //if(vr2s.find(iK0s->vertex()) != vr2s.end()) {
      if(vr2s.find(aVtx) != vr2s.end()) {
	//cout << "Found it in the collection." << endl;
      	std::vector< std::pair<TrackingVertexRef, double> > vVR 
	  = (std::vector< std::pair<TrackingVertexRef, double> >) vr2s[aVtx];
      }
    }
  }
*/
  //////////////////////////////
  // Do fake rate calculation //
  //////////////////////////////

  //cout << "Starting K0s fake rate calculation" << endl;
  // Kshorts
  double numK0sFound = 0.;
  double mass = 0.;
  std::vector<double> radDist;
  //  radDist.clear();
  //cout << "K0s collection size: " << k0sCollection->size() << endl;
  if ( k0sCollection->size() > 0 ) {
    //cout << "In loop" << endl;

    vector<reco::TrackRef> theDaughterTracks;
    for( reco::VertexCompositeCandidateCollection::const_iterator iK0s = k0sCollection->begin();
	 iK0s != k0sCollection->end();
	 iK0s++) {
      //cout << "In loop 2" << endl;
      // Fill mass of all K0S
      ksMassAll->Fill( iK0s->mass() );
      // Fill values to be histogrammed
      K0sCandpT = (sqrt( iK0s->momentum().perp2() ));
      K0sCandEta = iK0s->momentum().eta();
      K0sCandR = (sqrt( iK0s->vertex().perp2() ));
      K0sCandStatus = 0;
      //cout << "MASS" << endl;
      mass = iK0s->mass();
      //cout << "Pushing back daughters" << endl;

      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iK0s->daughter(0)) )).track() );
      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iK0s->daughter(1)) )).track() );
       
      //cout << "1" << endl;
      for (int itrack = 0; itrack < 2; itrack++) {
	K0sPiCandStatus[itrack] = 0;
      }

      std::vector< std::pair<TrackingParticleRef, double> > tp;
      TrackingParticleRef tpref;
      TrackingParticleRef firstDauTP;
      TrackingVertexRef k0sVtx;

      //cout << "2" << endl;
      // Loop through K0s candidate daugher tracks
      for(View<reco::Track>::size_type i=0; i<theDaughterTracks.size(); ++i){
	// Found track from theDaughterTracks
	RefToBase<reco::Track> track( theDaughterTracks.at(i) );
        
	//if(recSimColl.find(track) != recSimColl.end()) {
	if(recotosimCollectionH->find(track) != recotosimCollectionH->end()) {
	  //tp = recSimColl[track];
	  tp = (*recotosimCollectionH)[track];
	  if (tp.size() != 0) {
	    K0sPiCandStatus[i] = 1;
	    tpref = tp.begin()->first;

	    //if( simRecColl.find(tpref) == simRecColl.end() ) {
	    if( simtorecoCollectionH->find(tpref) == simtorecoCollectionH->end() ) {
	      K0sPiCandStatus[i] = 3;
	    }
	    //cout << "3" << endl;
	    TrackingVertexRef parentVertex = tpref->parentVertex();
	    if(parentVertex.isNonnull()) radDist.push_back(parentVertex->position().R());
	     
	    if( parentVertex.isNonnull() ) {
	      if( k0sVtx.isNonnull() ) {
		if( k0sVtx->position() == parentVertex->position() ) {
		  if( parentVertex->nDaughterTracks() == 2 ) {
		    if( parentVertex->nSourceTracks() == 0 ) {
		      // No source tracks found for K0s vertex; shouldn't happen, but does for evtGen events
		      K0sCandStatus = 6;
		    }
		    
		    for( TrackingVertex::tp_iterator iTP = parentVertex->sourceTracks_begin();
			 iTP != parentVertex->sourceTracks_end(); iTP++) {
		      if( (*iTP)->pdgId() == 310 ) {
			//cout << "4" << endl;
			K0sCandStatus = 1;
			realK0sFound++;
			numK0sFound += 1.;
			std::pair<TrackingParticleRef, TrackingParticleRef> pair(firstDauTP, tpref);
			// Pushing back a good V0
			trueK0s.push_back(pair);
			trueKsMasses.push_back(mass);
		      }
		      else {
			K0sCandStatus = 2;
			if( (*iTP)->pdgId() == 3122 ) {
			  K0sCandStatus = 7;
			}
		      }
		    }
		  }
		  else {
		    // Found a bad match because the mother has too many daughters
		    K0sCandStatus = 3;
		  }
		}
		else {
		  // Found a bad match because the parent vertices from the two tracks are different
		  K0sCandStatus = 4;
		}
	      }
	      else {
		// if k0sVtx is null, fill it with parentVertex to compare to the parentVertex from the second track
		k0sVtx = parentVertex;
		firstDauTP = tpref;
	      }
	    }//parent vertex is null
	  }//tp size zero
	}
	else {
	  //cout << "5" << endl;
	  K0sPiCandStatus[i] = 2;
	  noTPforK0sCand++;
	  K0sCandStatus = 5;
	  theDaughterTracks.clear();
	}
      }
      //cout << "6" << endl;
      theDaughterTracks.clear();
      // fill the fake rate histograms
      if( K0sCandStatus > 1 ) {
	//cout << "7" << endl;
	ksFakeVsR_num->Fill(K0sCandR);
	ksFakeVsEta_num->Fill(K0sCandEta);
	ksFakeVsPt_num->Fill(K0sCandpT);
	ksCandStatus->Fill((float) K0sCandStatus);
	fakeKsMass->Fill(mass);
	for( unsigned int ndx = 0; ndx < radDist.size(); ndx++ ) {
	  ksFakeDauRadDist->Fill(radDist[ndx]);
	}
      }
      if( K0sCandStatus == 5 ) {
	ksTkFakeVsR_num->Fill(K0sCandR);
	ksTkFakeVsEta_num->Fill(K0sCandEta);
	ksTkFakeVsPt_num->Fill(K0sCandpT);
      }
      ksFakeVsR_denom->Fill(K0sCandR);
      ksFakeVsEta_denom->Fill(K0sCandEta);
      ksFakeVsPt_denom->Fill(K0sCandpT);
    }
  }
  //cout << "Outside loop, why would it fail here?" << endl;
  //double numK0sFound = (double) realK0sFound;
  //cout << "numK0sFound: " << numK0sFound << endl;
  nKs->Fill( (float) numK0sFound );
  numK0sFound = 0.;

  //cout << "Starting Lambda fake rate calculation" << endl;

  double numLamFound = 0.;
  mass = 0.;
  radDist.clear();
  // Lambdas
  if ( lambdaCollection->size() > 0 ) {
    //cout << "In lam loop." << endl;
    
    vector<reco::TrackRef> theDaughterTracks;
    for( reco::VertexCompositeCandidateCollection::const_iterator iLam = lambdaCollection->begin();
	 iLam != lambdaCollection->end();
	 iLam++) {
      // Fill mass plot with ALL lambdas
      lamMassAll->Fill( iLam->mass() );
      // Fill values to be histogrammed
      LamCandpT = (sqrt( iLam->momentum().perp2() ));
      LamCandEta = iLam->momentum().eta();
      LamCandR = (sqrt( iLam->vertex().perp2() ));
      LamCandStatus = 0;
      mass = iLam->mass();
      
      //cout << "Lam daughter tracks" << endl;
      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iLam->daughter(0)) )).track() );
      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iLam->daughter(1)) )).track() );
      
      for (int itrack = 0; itrack < 2; itrack++) {
	LamPiCandStatus[itrack] = 0;
      }
      
      std::vector< std::pair<TrackingParticleRef, double> > tp;
      TrackingParticleRef tpref;
      TrackingParticleRef firstDauTP;
      TrackingVertexRef LamVtx;
      // Loop through Lambda candidate daughter tracks
      for(View<reco::Track>::size_type i=0; i<theDaughterTracks.size(); ++i){
	// Found track from theDaughterTracks
	//cout << "Looping over lam daughters" << endl;
	RefToBase<reco::Track> track( theDaughterTracks.at(i) );
	
	//if(recSimColl.find(track) != recSimColl.end()) {
	if(recotosimCollectionH->find(track) != recotosimCollectionH->end()) {
	  //tp = recSimColl[track];
	  tp = (*recotosimCollectionH)[track];
	  if (tp.size() != 0) {
	    LamPiCandStatus[i] = 1;
	    tpref = tp.begin()->first;

	    //if( simRecColl.find(tpref) == simRecColl.end() ) {
	    if( simtorecoCollectionH->find(tpref) == simtorecoCollectionH->end() ) {
	      LamPiCandStatus[i] = 3;
	    }
	    TrackingVertexRef parentVertex = tpref->parentVertex();
	    if( parentVertex.isNonnull() ) radDist.push_back(parentVertex->position().R());
	     
	    if( parentVertex.isNonnull() ) {
	      if( LamVtx.isNonnull() ) {
		if( LamVtx->position() == parentVertex->position() ) {
		  if( parentVertex->nDaughterTracks() == 2 ) {
		    if( parentVertex->nSourceTracks() == 0 ) {
		      // No source tracks found for K0s vertex; shouldn't happen, but does for evtGen events
		      LamCandStatus = 6;
		    }

		    for( TrackingVertex::tp_iterator iTP = parentVertex->sourceTracks_begin();
			 iTP != parentVertex->sourceTracks_end(); ++iTP) {
		      if( abs((*iTP)->pdgId()) == 3122 ) {
			LamCandStatus = 1;
			realLamFound++;
			numLamFound += 1.;
			std::pair<TrackingParticleRef, TrackingParticleRef> pair(firstDauTP, tpref);
			// Pushing back a good V0
			trueLams.push_back(pair);
			trueLamMasses.push_back(mass);
		      }
		      else {
			LamCandStatus = 2;
			if( abs((*iTP)->pdgId() ) == 310 ) {
			  LamCandStatus = 7;
			}
		      }
		      //if(iTP != parentVertex->sourceTracks_end()) {
		      //cout << "Bogus check 1" << endl;
		      //}
		    }
		  }
		  else {
		    // Found a bad match because the mother has too many daughters
		    LamCandStatus = 3;
		  }
		}
		else {
		  // Found a bad match because the parent vertices from the two tracks are different
		  LamCandStatus = 4;
		}
	      }
	      else {
		// if lamVtx is null, fill it with parentVertex to compare to the parentVertex from the second track
		LamVtx = parentVertex;
		firstDauTP = tpref;
	      }
	    }//parent vertex is null
	  }//tp size zero
	}
	else {
	  LamPiCandStatus[i] = 2;
	  noTPforLamCand++;
	  LamCandStatus = 5;
	  theDaughterTracks.clear();
	}
      }
      theDaughterTracks.clear();
      // fill the fake rate histograms
      //cout << "Fill lam fake rate histos" << endl;
      if( LamCandStatus > 1 ) {
	//cout << "fake 1" << endl;
	//cout << "fake 1.5" << endl;
	lamFakeVsR_num->Fill(LamCandR);
	//cout << "fake 2" << endl;
	lamFakeVsEta_num->Fill(LamCandEta);
	//cout << "fake 3" << endl;
	lamFakeVsPt_num->Fill(LamCandpT);
	//cout << "fake 4" << endl;
	lamCandStatus->Fill((float) LamCandStatus);
	//cout << "fake 5" << endl;
	fakeLamMass->Fill(mass);
	//cout << "fake 6" << endl;
	for( unsigned int ndx = 0; ndx < radDist.size(); ndx++ ) {
	  lamFakeDauRadDist->Fill(radDist[ndx]);
	}
      }
      //cout << "Fill lam Tk fake histos" << endl;
      if( K0sCandStatus == 5 ) {
	lamTkFakeVsR_num->Fill(LamCandR);
	lamTkFakeVsEta_num->Fill(LamCandEta);
	lamTkFakeVsPt_num->Fill(LamCandpT);
      }
      //cout << "Fill denominators" << endl;
      lamFakeVsR_denom->Fill(LamCandR);
      lamFakeVsEta_denom->Fill(LamCandEta);
      lamFakeVsPt_denom->Fill(LamCandpT);
    }
  }
  //cout << "Filling numLamFound" << endl;
  nLam->Fill( (double) numLamFound );
  numLamFound = 0.;


  ///////////////////////////////
  // Do efficiency calculation //
  ///////////////////////////////

  //cout << "Starting Lambda efficiency" << endl;
  // Lambdas

  for(TrackingParticleCollection::size_type i = 0; i < tPCeff.size(); i++) {
    TrackingParticleRef tpr1(TPCollectionEff, i);
    const TrackingParticle* itp1 = tpr1.get();
    if( (itp1->pdgId() == 211 || itp1->pdgId() == 2212)
	&& itp1->parentVertex().isNonnull()
	&& abs(itp1->momentum().eta()) < 2.4
	&& sqrt( itp1->momentum().perp2() ) > 0.9) {
      bool isLambda = false;
      if( itp1->pdgId() == 2212 ) isLambda = true;
      if( itp1->parentVertex()->nDaughterTracks() == 2 ) {

	TrackingVertexRef piCand1Vertex = itp1->parentVertex();
	for(TrackingVertex::tp_iterator iTP1 = piCand1Vertex->sourceTracks_begin();
	    iTP1 != piCand1Vertex->sourceTracks_end(); iTP1++) {
	  if( abs((*iTP1)->pdgId()) == 3122 ) {
	    //double motherpT = (*iTP1)->pt();
	    //	     ----->>>>>>Keep going here
	    for(TrackingParticleCollection::size_type j=0;
		j < tPCeff.size();
		j++) {
	      TrackingParticleRef tpr2(TPCollectionEff, j);
	      const TrackingParticle* itp2 = tpr2.get();
	      int particle2pdgId;
	      if (isLambda) particle2pdgId = -211;
	      else particle2pdgId = -2212;
	      if( itp2->pdgId() == particle2pdgId
		  && itp2->parentVertex().isNonnull()
		  && abs(itp2->momentum().eta()) < 2.4
		  && sqrt(itp2->momentum().perp2()) > 0.9) {
		if(itp2->parentVertex() == itp1->parentVertex()) {
		  // Found a good pair of Lambda daughters
		  TrackingVertexRef piCand2Vertex = itp2->parentVertex();
		  for (TrackingVertex::tp_iterator iTP2 = piCand2Vertex->sourceTracks_begin();
		       iTP2 != piCand2Vertex->sourceTracks_end(); 
		       ++iTP2) {
		    LamGenEta = LamGenpT = LamGenR = 0.;
		    LamGenStatus = 0;
		    for(int ifill = 0;
			ifill < 2;
			ifill++) {
		      // do nothing?
		    }
		    if( abs((*iTP2)->pdgId()) == 3122 ) {
		      // found generated Lambda
		      
		      LamGenpT = sqrt((*iTP2)->momentum().perp2());
		      LamGenEta = (*iTP2)->momentum().eta();
		      LamGenR = sqrt(itp2->vertex().perp2());
		      genLam++;
		      if(trueLams.size() > 0) {
			int loop_1 = 0;
			for(std::vector< pair<TrackingParticleRef, TrackingParticleRef> >::const_iterator iEffCheck = trueLams.begin();
			    iEffCheck != trueLams.end();
			    iEffCheck++) {
			  //cout << "In LOOP" << endl;
			  if( itp1->parentVertex() == iEffCheck->first->parentVertex()
			      && itp2->parentVertex() == iEffCheck->second->parentVertex() ) {
			    realLamFoundEff++;
			    //V0Producer found the generated Lambda
			    LamGenStatus = 1;
			    //cout << "Maybe it's here.." << endl;
			    goodLamMass->Fill(trueLamMasses[loop_1]);
			    //cout << "Did we make it?" << endl;
			    break;
			  }
			  else {
			    //V0Producer didn't find the generated Lambda
			    LamGenStatus = 2;
			  }
			  loop_1++;
			}
		      }
		      else {
			//No V0 cand found, so V0Producer didn't find the generated Lambda
			LamGenStatus = 2;
		      }
		      std::vector< std::pair<RefToBase<reco::Track>, double> > rt1;
		      std::vector< std::pair<RefToBase<reco::Track>, double> > rt2;
		      
		      //if( simRecColl.find(tpr1) != simRecColl.end() ) {
		      if( simtorecoCollectionH->find(tpr1) != simtorecoCollectionH->end() ) {
			//rt1 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) simRecColl[tpr1];
			rt1 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr1];
			if(rt1.size() != 0) {
			  LamPiEff[0] = 1; //Found the first daughter track
			  edm::RefToBase<reco::Track> t1 = rt1.begin()->first;
			}
		      }
		      else {
			LamPiEff[0] = 2;//First daughter not found
		      }
		      //if( (simRecColl.find(tpr2) != simRecColl.end()) ) {
		      if( (simtorecoCollectionH->find(tpr2) != simtorecoCollectionH->end()) ) {
			//rt2 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) simRecColl[tpr2];
			rt2 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr2];
			if(rt2.size() != 0) {
			  LamPiEff[1] = 1;//Found the second daughter track
			  edm::RefToBase<reco::Track> t2 = rt2.begin()->first;
			}
		      }
		      else {
			LamPiEff[1] = 2;//Second daughter not found
		      }
		      
		      if( LamGenStatus == 1
			  && (LamPiEff[0] == 2 || LamPiEff[1] == 2) ) {
			// Good Lambda found, but recoTrack->trackingParticle->recoTrack didn't work
			LamGenStatus = 4;
			realLamFoundEff--;
		      }
		      if( LamGenStatus == 2
			  && (LamPiEff[0] == 2 || LamPiEff[1] == 2) ) {
			// Lambda not found because we didn't find a daughter track
			LamGenStatus = 3;
		      }
		      //cout << "LamGenStatus: " << LamGenStatus << ", LamPiEff[i]: " << LamPiEff[0] << ", " << LamPiEff[1] << endl;
		      // Fill histograms
		      if(LamGenR > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsR_num->Fill(LamGenR);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsR_num->Fill(LamGenR);
			}
			lamEffVsR_denom->Fill(LamGenR);
		      }
		      if(abs(LamGenEta) > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsEta_num->Fill(LamGenEta);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsEta_num->Fill(LamGenEta);
			}
			lamEffVsEta_denom->Fill(LamGenEta);
		      }
		      if(LamGenpT > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsPt_num->Fill(LamGenpT);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsPt_num->Fill(LamGenpT);
			}
			lamEffVsPt_denom->Fill(LamGenpT);
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  //Kshorts

  //cout << "Starting Kshort efficiency" << endl;
  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
    TrackingParticleRef tpr1(TPCollectionEff, i);
    const TrackingParticle* itp1 = tpr1.get();
    // only count the efficiency for pions with |eta|<2.4 and pT>0.9 GeV. First search for a suitable pi+
    if ( itp1->pdgId() == 211 
	 && itp1->parentVertex().isNonnull() 
	 && abs(itp1->momentum().eta()) < 2.4 
	 && sqrt(itp1->momentum().perp2()) > 0.9) {
      if ( itp1->parentVertex()->nDaughterTracks() == 2 ) {
	TrackingVertexRef piCand1Vertex = itp1->parentVertex();	       
	//check trackingParticle pion for a Ks mother
	for (TrackingVertex::tp_iterator iTP1 = piCand1Vertex->sourceTracks_begin();
	     iTP1 != piCand1Vertex->sourceTracks_end(); ++iTP1) {
	  //iTP1 is a TrackingParticle
	  if ( (*iTP1)->pdgId()==310 ) {
	    //with a Ks mother found for the pi+, loop through trackingParticles again to find a pi-
	    for (TrackingParticleCollection::size_type j=0; j<tPCeff.size(); j++){
	      TrackingParticleRef tpr2(TPCollectionEff, j);
	      const TrackingParticle* itp2 = tpr2.get();
	      
	      if ( itp2->pdgId() == -211 && itp2->parentVertex().isNonnull()  
		   && abs(itp2->momentum().eta()) < 2.4 
		   && sqrt(itp2->momentum().perp2()) > 0.9) {
		//check the pi+ and pi- have the same vertex
		if ( itp2->parentVertex() == itp1->parentVertex() ) {
		  TrackingVertexRef piCand2Vertex = itp2->parentVertex();	       
		  for (TrackingVertex::tp_iterator iTP2 = piCand2Vertex->sourceTracks_begin();
		       iTP2 != piCand2Vertex->sourceTracks_end(); ++iTP2) {
		    //iTP2 is a TrackingParticle
		    K0sGenEta = K0sGenpT = K0sGenR = 0.;
		    K0sGenStatus = 0;
		    if( (*iTP2)->pdgId() == 310 ) {
		      K0sGenpT = sqrt( (*iTP2)->momentum().perp2() );
		      K0sGenEta = (*iTP2)->momentum().eta();
		      K0sGenR = sqrt(itp2->vertex().perp2());
		      genK0s++;
		      int loop_2 = 0;
		      if( trueK0s.size() > 0 ) {
			for( std::vector< pair<TrackingParticleRef, TrackingParticleRef> >::const_iterator iEffCheck = trueK0s.begin();
			     iEffCheck != trueK0s.end();
			     iEffCheck++) {
			  //if the parent vertices for the tracks are the same, then the generated Ks was found
			  if (itp1->parentVertex()==iEffCheck->first->parentVertex() &&
			      itp2->parentVertex()==iEffCheck->second->parentVertex())  {
			    realK0sFoundEff++;
			    K0sGenStatus = 1;
			    //cout << "Maybe here?" << endl;
			    goodKsMass->Fill(trueKsMasses[loop_2]);
			    //cout << "We made it...." << endl;
			    break;
			  }
			  else {
			    K0sGenStatus = 2;
			  }
			}
		      }
		      else {
			K0sGenStatus = 2;
		      }

		      // Check if the generated Ks tracks were found or not
		      // by searching the recoTracks list for a match to the trackingParticles

		      std::vector<std::pair<RefToBase<reco::Track>, double> > rt1;
		      std::vector<std::pair<RefToBase<reco::Track>, double> > rt2;
		      
		      //if( simRecColl.find(tpr1) != simRecColl.end() ) {
		      if( simtorecoCollectionH->find(tpr1) != simtorecoCollectionH->end() ) {
			rt1 = (std::vector< std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr1];
//simRecColl[tpr1];
			if(rt1.size() != 0) {
			  //First pion found
			  K0sPiEff[0] = 1;
			  edm::RefToBase<reco::Track> t1 = rt1.begin()->first;
			}
		      }
		      else {
			//First pion not found
			K0sPiEff[0] = 2;
		      }
		      
		      //if( simRecColl.find(tpr2) != simRecColl.end() ) {
		      if( simtorecoCollectionH->find(tpr2) != simtorecoCollectionH->end() ) {
			rt2 = (std::vector< std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr2];
//simRecColl[tpr2];
			if(rt2.size() != 0) {
			  //Second pion found
			  K0sPiEff[1] = 1;
			  edm::RefToBase<reco::Track> t2 = rt2.begin()->first;
			}
		      }
		      else {
			K0sPiEff[1] = 2;
		      }
		      //cout << "Status: " << K0sGenStatus << ", K0sPiEff[i]: " << K0sPiEff[0] << ", " << K0sPiEff[1] << endl;
		      if(K0sGenStatus == 1
			 && (K0sPiEff[0] == 2 || K0sPiEff[1] == 2)) {
			K0sGenStatus = 4;
			realK0sFoundEff--;
		      }
		      if(K0sGenStatus == 2
			 && (K0sPiEff[0] == 2 || K0sPiEff[1] == 2)) {
			K0sGenStatus = 3;
		      }
		      if(K0sPiEff[0] == 1 && K0sPiEff[1] == 1) {
			k0sTracksFound++;
		      }
		      //Fill Histograms
		      if(K0sGenR > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsR_num->Fill(K0sGenR);
			}
			if((double) K0sGenStatus < 2.5) {			  
			  ksTkEffVsR_num->Fill(K0sGenR);
			}
			ksEffVsR_denom->Fill(K0sGenR);
		      }
		      if(abs(K0sGenEta) > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsEta_num->Fill(K0sGenEta);
			}
			if((double) K0sGenStatus < 2.5) {
			  ksTkEffVsEta_num->Fill(K0sGenEta);
			}
			ksEffVsEta_denom->Fill(K0sGenEta);
		      }
		      if(K0sGenpT > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsPt_num->Fill(K0sGenpT);
			}
			if((double) K0sGenStatus < 2.5) {
			  ksTkEffVsPt_num->Fill(K0sGenpT);
			}
			ksEffVsPt_denom->Fill(K0sGenpT);
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  delete thePrimary;
}

void V0Validator::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  //theDQMstore->showDirStructure();
  if(theDQMRootFileName.size() && theDQMstore) {
    theDQMstore->save(theDQMRootFileName);
  }
}


//void V0Validator::endJob() {
  //std::cout << "In endJob()" << std::endl;
  /*ksEffVsRHist->Divide(ksEffVsRHist_denom);
  ksEffVsEtaHist->Divide(ksEffVsEtaHist_denom);
  ksEffVsPtHist->Divide(ksEffVsPtHist_denom);
  ksTkEffVsRHist->Divide(ksEffVsRHist_denom);
  ksTkEffVsEtaHist->Divide(ksEffVsEtaHist_denom);
  ksTkEffVsPtHist->Divide(ksEffVsPtHist_denom);
  ksFakeVsRHist->Divide(ksFakeVsRHist_denom);
  ksFakeVsEtaHist->Divide(ksFakeVsEtaHist_denom);
  ksFakeVsPtHist->Divide(ksFakeVsPtHist_denom);
  ksTkFakeVsRHist->Divide(ksFakeVsRHist_denom);
  ksTkFakeVsEtaHist->Divide(ksFakeVsEtaHist_denom);
  ksTkFakeVsPtHist->Divide(ksFakeVsPtHist_denom);

  lamEffVsRHist->Divide(lamEffVsRHist_denom);
  lamEffVsEtaHist->Divide(lamEffVsEtaHist_denom);
  lamEffVsPtHist->Divide(lamEffVsPtHist_denom);
  lamTkEffVsRHist->Divide(lamEffVsRHist_denom);
  lamTkEffVsEtaHist->Divide(lamEffVsEtaHist_denom);
  lamTkEffVsPtHist->Divide(lamEffVsPtHist_denom);
  lamFakeVsRHist->Divide(lamFakeVsRHist_denom);
  lamFakeVsEtaHist->Divide(lamFakeVsEtaHist_denom);
  lamFakeVsPtHist->Divide(lamFakeVsPtHist_denom);
  lamTkFakeVsRHist->Divide(lamFakeVsRHist_denom);
  lamTkFakeVsEtaHist->Divide(lamFakeVsEtaHist_denom);
  lamTkFakeVsPtHist->Divide(lamFakeVsPtHist_denom);

  theDQMstore->cd();
  std::string subDirName = dirName + "/Efficiency";
  theDQMstore->setCurrentFolder(subDirName.c_str());

  ksEffVsR = theDQMstore->book1D("KsEffVsR", ksEffVsRHist);
  ksEffVsEta = theDQMstore->book1D("KsEffVsEta", ksEffVsEtaHist);
  ksEffVsPt = theDQMstore->book1D("KsEffVsPt", ksEffVsPtHist);
  ksTkEffVsR = theDQMstore->book1D("KsTkEffVsR", ksTkEffVsRHist);
  ksTkEffVsEta = theDQMstore->book1D("KsTkEffVsEta", ksTkEffVsEtaHist);
  ksTkEffVsPt = theDQMstore->book1D("KsTkEffVsPt", ksTkEffVsPtHist);

  lamEffVsR = theDQMstore->book1D("LamEffVsR", lamEffVsRHist);
  lamEffVsEta = theDQMstore->book1D("LamEffVsEta", lamEffVsEtaHist);
  lamEffVsPt = theDQMstore->book1D("LamEffVsPt", lamEffVsPtHist);
  lamTkEffVsR = theDQMstore->book1D("LamTkEffVsR", lamTkEffVsRHist);
  lamTkEffVsEta = theDQMstore->book1D("LamTkEffVsEta", lamTkEffVsEtaHist);
  lamTkEffVsPt = theDQMstore->book1D("LamTkEffVsPt", lamTkEffVsPtHist);

  theDQMstore->cd();
  subDirName = dirName + "/Fake";
  theDQMstore->setCurrentFolder(subDirName.c_str());

  ksFakeVsR = theDQMstore->book1D("KsFakeVsR", ksFakeVsRHist);
  ksFakeVsEta = theDQMstore->book1D("KsFakeVsEta", ksFakeVsEtaHist);
  ksFakeVsPt = theDQMstore->book1D("KsFakeVsPt", ksFakeVsPtHist);
  ksTkFakeVsR = theDQMstore->book1D("KsTkFakeVsR", ksTkFakeVsRHist);
  ksTkFakeVsEta = theDQMstore->book1D("KsTkFakeVsEta", ksTkFakeVsEtaHist);
  ksTkFakeVsPt = theDQMstore->book1D("KsTkFakeVsPt", ksTkFakeVsPtHist);

  lamFakeVsR = theDQMstore->book1D("LamFakeVsR", lamFakeVsRHist);
  lamFakeVsEta = theDQMstore->book1D("LamFakeVsEta", lamFakeVsEtaHist);
  lamFakeVsPt = theDQMstore->book1D("LamFakeVsPt", lamFakeVsPtHist);
  lamTkFakeVsR = theDQMstore->book1D("LamTkFakeVsR", lamTkFakeVsRHist);
  lamTkFakeVsEta = theDQMstore->book1D("LamTkFakeVsEta", lamTkFakeVsEtaHist);
  lamTkFakeVsPt = theDQMstore->book1D("LamTkFakeVsPt", lamTkFakeVsPtHist);

  nKs = theDQMstore->book1D("nK0s", nKsHist);
  nLam = theDQMstore->book1D("nLam", nLamHist);

  ksCandStatusME = theDQMstore->book1D("ksCandStatus", ksCandStatusHist);
  lamCandStatusME = theDQMstore->book1D("lamCandStatus", lamCandStatusHist);

  fakeKsMass = theDQMstore->book1D("ksMassFake", fakeKsMassHisto);
  goodKsMass = theDQMstore->book1D("ksMassGood", goodKsMassHisto);
  fakeLamMass = theDQMstore->book1D("lamMassFake", fakeLamMassHisto);
  goodLamMass = theDQMstore->book1D("lamMassGood", goodLamMassHisto);

  ksFakeDauRadDist = theDQMstore->book1D("radDistFakeKs", ksFakeDauRadDistHisto);
  lamFakeDauRadDist = theDQMstore->book1D("radDistFakeLam", lamFakeDauRadDistHisto);

  // ***************************************/
  /*theDQMstore->tag(ksEffVsR->getFullname(), 1);
  theDQMstore->tag(ksEffVsEta->getFullname(), 2);
  theDQMstore->tag(ksEffVsPt->getFullname(), 3);
  theDQMstore->tag(ksTkEffVsR->getFullname(), 4);
  theDQMstore->tag(ksTkEffVsEta->getFullname(), 5);
  theDQMstore->tag(ksTkEffVsPt->getFullname(), 6);

  theDQMstore->tag(lamEffVsR->getFullname(), 7);
  theDQMstore->tag(lamEffVsEta->getFullname(), 8);
  theDQMstore->tag(lamEffVsPt->getFullname(), 9);
  theDQMstore->tag(lamTkEffVsR->getFullname(), 10);
  theDQMstore->tag(lamTkEffVsEta->getFullname(), 11);
  theDQMstore->tag(lamTkEffVsPt->getFullname(), 12);

  theDQMstore->tag(ksFakeVsR->getFullname(), 13);
  theDQMstore->tag(ksFakeVsEta->getFullname(), 14);
  theDQMstore->tag(ksFakeVsPt->getFullname(), 15);
  theDQMstore->tag(ksTkFakeVsR->getFullname(), 16);
  theDQMstore->tag(ksTkFakeVsEta->getFullname(), 17);
  theDQMstore->tag(ksTkFakeVsPt->getFullname(), 18);

  theDQMstore->tag(lamFakeVsR->getFullname(), 19);
  theDQMstore->tag(lamFakeVsEta->getFullname(), 20);
  theDQMstore->tag(lamFakeVsPt->getFullname(), 21);
  theDQMstore->tag(lamTkFakeVsR->getFullname(), 22);
  theDQMstore->tag(lamTkFakeVsEta->getFullname(), 23);
  theDQMstore->tag(lamTkFakeVsPt->getFullname(), 24);

  theDQMstore->tag(nKs->getFullname(), 25);
  theDQMstore->tag(nLam->getFullname(), 26);
  
  theDQMstore->tag(ksCandStatusME->getFullname(), 27);
  theDQMstore->tag(lamCandStatusME->getFullname(), 28);

  theDQMstore->tag(fakeKsMass->getFullname(), 29);
  theDQMstore->tag(goodKsMass->getFullname(), 30);
  theDQMstore->tag(fakeLamMass->getFullname(), 31);
  theDQMstore->tag(goodLamMass->getFullname(), 32);

  theDQMstore->tag(ksFakeDauRadDist->getFullname(), 33);
  theDQMstore->tag(lamFakeDauRadDist->getFullname(), 34);*/
  /****************************************/

  /*theDQMstore->showDirStructure();
    theDQMstore->save(theDQMRootFileName);*/
//}

//define this as a plug-in
//DEFINE_FWK_MODULE(V0Validator);
