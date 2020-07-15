#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePropagator.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
using namespace GeomDetEnumerators;

#include <iostream>

using namespace std;
using namespace edm;

#include "TFile.h"
#include "TH1D.h"
#include "TProfile.h"

template <class T>
T sqr(T t) {
  return t * t;
}

class TestMS : public edm::EDAnalyzer {
public:
  explicit TestMS(const edm::ParameterSet& conf);
  ~TestMS() override;
  void beginRun(edm::Run const& run, const edm::EventSetup& es) override;
  void analyze(const edm::Event& ev, const edm::EventSetup& es) override;

private:
  TFile* rootFile;
  TH1F *hB1, *hB2, *hB3, *hF1, *hF2;
};

TestMS::TestMS(const edm::ParameterSet& conf) {
  rootFile = new TFile("histos.root", "RECREATE");
  const int nch = 250;
  const float rmin = 0;
  const float zmin = 0;
  const float zmax = 25;
  const float rmax = 15;

  edm::LogInfo("TestMS") << " CRATEING HISTOS";
  hF1 = new TH1F("hSigmaF1", "hSigmaF1", nch, rmin, rmax);
  hF2 = new TH1F("hSigmaF2", "hSigmaF2", nch, rmin, rmax);
  hB1 = new TH1F("hSigmaB1", "hSigmaB1", nch, zmin, zmax);
  hB2 = new TH1F("hSigmaB2", "hSigmaB2", nch, zmin, zmax);
  hB3 = new TH1F("hSigmaB3", "hSigmaB3", nch, zmin, zmax);
}

TestMS::~TestMS() {
  edm::LogInfo("TestMS") << " DTOR";
  std::cout << "WRITING ROOT FILE" << std::endl;
  rootFile->Write();
  std::cout << "rootFile WRITTEN" << std::endl;
}

void TestMS::analyze(const edm::Event& ev, const edm::EventSetup& es) {}

void TestMS::beginRun(edm::Run const& run, const edm::EventSetup& es) {
  edm::ESHandle<GeometricSearchTracker> tracker;
  es.get<TrackerRecoGeometryRecord>().get(tracker);

  vector<BarrelDetLayer const*> barrel = tracker->barrelLayers();
  //  vector<ForwardDetLayer*> endcap=tracker->posForwardLayers();
  vector<ForwardDetLayer const*> endcap = tracker->negForwardLayers();

  MultipleScatteringParametrisation sb1(barrel[0], es);
  MultipleScatteringParametrisation sb2(barrel[1], es);
  MultipleScatteringParametrisation sb3(barrel[2], es);

  MultipleScatteringParametrisation sf1(endcap[0], es);
  MultipleScatteringParametrisation sf2(endcap[1], es);

  const int nch = 250;
  const float rmin = 0;
  const float zmin = 0;
  const float zmax = 25;
  const float rmax = 15;
  float r1 = 4.43;
  float r2 = 7.34;
  float r3 = 10.2;
  float z1 = 34.5;
  float z2 = 46.5;  // float z3 = 58.5;
  z1 = -z1;
  z2 = -z2;  // z3 = -z3;

  cout << "HERE !!!! " << endl;
  for (int i = 0; i < nch; i++) {
    float r = rmin + (i + 0.5) * (rmax - rmin) / nch;
    float z = zmin + (i + 0.5) * (zmax - zmin) / nch;
    float cotThetaB1 = z / r1;
    float cotThetaB2 = z / r2;
    float cotThetaB3 = z / r3;
    float cotThetaF1 = z1 / r;
    float cotThetaF2 = z2 / r;
    float msB1 = sb1(1., cotThetaB1);
    float msB2 = sb2(1., cotThetaB2);
    float msB3 = sb3(1., cotThetaB3);
    float msF1 = sf1(1., cotThetaF1);
    float msF2 = sf2(1., cotThetaF2);
    //    std::cout <<"--------------------->"
    //            <<i<<" r="<<r<<" cot="<<cotThetaF2<<" ms="<<msF1<<" "<<msF2<<std::endl;
    //
    hB1->Fill(fabs(z), msB1);
    hB2->Fill(fabs(z), msB2);
    hB3->Fill(fabs(z), msB3);
    hF1->Fill(r, msF1);
    hF2->Fill(r, msF2);
  }
}

DEFINE_FWK_MODULE(TestMS);
