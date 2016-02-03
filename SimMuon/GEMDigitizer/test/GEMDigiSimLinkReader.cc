#ifndef SimMuon_GEMDigiSimLinkReader_h
#define SimMuon_GEMDigiSimLinkReader_h

/** \class GEMDigiSimLinkReader
 *
 *  Dumps GEM digis
 *
 *  \authors: Roumyana Hadjiiska
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"
#include <map>
#include <set>

#include "DataFormats/Common/interface/DetSet.h"
#include <iostream>

#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;

class GEMDigiSimLinkReader: public edm::EDAnalyzer
{
public:

  explicit GEMDigiSimLinkReader(const edm::ParameterSet& pset);

  virtual ~GEMDigiSimLinkReader()
  {
  }

  void analyze(const edm::Event &, const edm::EventSetup&);

private:

//  string label;

  edm::EDGetTokenT<edm::PSimHitContainer> simhitToken_;
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken_;
  edm::EDGetTokenT<edm::DetSetVector<GEMDigiSimLink> > gemDigiSimLinkToken_;


  TH1F *hProces;
  TH1F *hParticleTypes;
  TH1F *hAllSimHitsType;

};

GEMDigiSimLinkReader::GEMDigiSimLinkReader(const edm::ParameterSet& pset) :
  simhitToken_(consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("simhitToken"))),
  gemDigiToken_(consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("gemDigiToken"))),
  gemDigiSimLinkToken_(consumes<edm::DetSetVector<GEMDigiSimLink> >(pset.getParameter<edm::InputTag>("gemDigiSimLinkToken")))
{
//  label = pset.getUntrackedParameter<string> ("label", "simMuonGEMDigis");

  edm::Service<TFileService> fs;

  hProces = fs->make<TH1F>("hProces", "Process type for all the simHits", 20, 0, 20);
  hAllSimHitsType = fs->make<TH1F>("hAllSimHitsType", "pdgId for All simHits", 500, 0, 500);
  hParticleTypes = fs->make<TH1F>("hParticleTypes", "pdgId for digitized simHits", 500, 0, 500);

}


/*
GEMDigiSimLinkReader::GEMDigiSimLinkReader(const edm::ParameterSet& pset)
{
  label = pset.getUntrackedParameter<string> ("label", "simMuonGEMDigis");

  edm::Service<TFileService> fs;

  hProces = fs->make<TH1F>("hProces", "Process type for all the simHits", 20, 0, 20);
  hAllSimHitsType = fs->make<TH1F>("hAllSimHitsType", "pdgId for All simHits", 500, 0, 500);
  hParticleTypes = fs->make<TH1F>("hParticleTypes", "pdgId for digitized simHits", 500, 0, 500);

}
*/
void GEMDigiSimLinkReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  //  cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::Handle<GEMDigiCollection> digis;
 // event.getByLabel(label, digis);
  event.getByToken(gemDigiToken_, digis);

  edm::Handle<edm::PSimHitContainer> simHits;
//  event.getByLabel("g4SimHits", "MuonGEMHits", simHits);
  event.getByToken(simhitToken_, simHits);    

  edm::ESHandle<GEMGeometry> pDD;
  eventSetup.get<MuonGeometryRecord> ().get(pDD);

//  edm::Handle<edm::DetSetVector<StripDigiSimLink> > thelinkDigis;
//  event.getByLabel(label, "GEM", thelinkDigis);

//  edm::Handle<edm::DetSetVector<GEMDigiSimLink> > theSimlinkDigis;
//  event.getByLabel(label, "GEM", theSimlinkDigis);
  edm::Handle< edm::DetSetVector<GEMDigiSimLink> > theSimlinkDigis;
  event.getByToken(gemDigiSimLinkToken_, theSimlinkDigis);


  //loop over all simhits
  for(const auto& simHit: *simHits)
  {
    std::cout << "particle type\t" << simHit.particleType()
              << "\tprocess type\t" << simHit.processType()
              << std::endl;

    hProces->Fill(simHit.processType());
    hAllSimHitsType->Fill(simHit.particleType());

  }

//loop over the detectors which have digitized simhits
  for (edm::DetSetVector<GEMDigiSimLink>::const_iterator itsimlink = theSimlinkDigis->begin(); itsimlink
      != theSimlinkDigis->end(); itsimlink++)
  {
    //get the particular detector
    int detid = itsimlink->detId();
    const GEMEtaPartition* roll = pDD->etaPartition(detid);
    const GEMDetId gemId = roll->id();
//    const StripTopology& topology = roll->specificTopology();
    const int nstrips = roll->nstrips();

//loop over GemDigiSimLinks
    for (edm::DetSet<GEMDigiSimLink>::const_iterator link_iter = itsimlink->data.begin(); link_iter
        != itsimlink->data.end(); ++link_iter)
    {
      std::cout << "roll Id\t" << gemId << std::endl;
      std::cout << "number of strips \t" << nstrips << std::endl;
      int strip = link_iter->getStrip();
      int processtype = link_iter->getProcessType();
      int particletype = link_iter->getParticleType();
      int bx = link_iter->getBx();
      double myEnergyLoss = link_iter->getEnergyLoss();

      std::cout << "simhit particle type\t" << particletype << std::endl;
      std::cout << "simhit process type\t" << processtype << std::endl;
      std::cout << "linked to strip with number\t" << strip << std::endl;
      std::cout << "in bunch crossing\t" << bx << std::endl;
      std::cout << "\tenergy loss\t" << myEnergyLoss << std::endl;

      hParticleTypes->Fill(particletype);
    }
  }//end given detector
}

#endif
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE( GEMDigiSimLinkReader);
