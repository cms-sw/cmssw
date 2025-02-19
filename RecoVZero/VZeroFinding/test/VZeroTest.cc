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

#include "DataFormats/VZero/interface/VZero.h"
#include "DataFormats/VZero/interface/VZeroFwd.h"

#include <iostream>

using namespace std;

// ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"

/*****************************************************************************/
class VZeroTest : public edm::EDAnalyzer
{
 public:
   explicit VZeroTest(const edm::ParameterSet& pset);
   ~VZeroTest();
   virtual void beginJob() { }
   virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
   virtual void endJob() { }

 private:
   TNtuple * ntuple;
   TFile * resultFile;
};

/*****************************************************************************/
VZeroTest::VZeroTest(const edm::ParameterSet& pset)
{
  edm::LogInfo("VZeroTest") << " constructor";

  string resultName = pset.getParameter<string>("resultName");
  resultFile = new TFile(resultName.c_str(),"RECREATE");
  resultFile->cd();
  ntuple = new TNtuple("vzero","vzero","bpos:bneg:dcar:dcaz:r:d:b:pt:alpha");
}

/*****************************************************************************/
VZeroTest::~VZeroTest()
{
  edm::LogInfo("VZeroTest") << " destructor";

  resultFile->cd();
  ntuple->Write();
  resultFile->Close();
}

/*****************************************************************************/
void VZeroTest::analyze(
    const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<reco::VZeroCollection> vZeroCollection;
  ev.getByLabel("pixelVZeros",vZeroCollection);
  const reco::VZeroCollection vZeros = *(vZeroCollection.product());

  for(reco::VZeroCollection::const_iterator it = vZeros.begin();
                                            it!= vZeros.end(); it++)
  {
    vector<float> result; 

    result.push_back(it->positiveDaughter()->d0());
    result.push_back(it->negativeDaughter()->d0());
    result.push_back(it->dca());
    result.push_back(it->dca());
    result.push_back(it->crossingPoint().Rho());
    result.push_back(it->crossingPoint().R());
    result.push_back(it->impactMother());
    result.push_back(it->armenterosPt());
    result.push_back(it->armenterosAlpha());

    ntuple->Fill(&result[0]);
  } 
}

DEFINE_FWK_MODULE(VZeroTest);
