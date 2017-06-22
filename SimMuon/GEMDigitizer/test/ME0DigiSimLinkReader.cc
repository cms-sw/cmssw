/** \class ME0DigiSimLinkReader
 *
 *  Dumps ME0DigiSimLinks digis
 *
 *  \authors: Roumyana Hadjiiska
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/GEMDigiSimLink/interface/ME0DigiSimLink.h"
#include <map>
#include <set>
#include "DataFormats/Common/interface/DetSet.h"
#include <iostream>
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

using namespace std;

class ME0DigiSimLinkReader: public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:

  explicit ME0DigiSimLinkReader(const edm::ParameterSet& pset);

  virtual ~ME0DigiSimLinkReader()
  {
  }

  void analyze(const edm::Event &, const edm::EventSetup&);

private:

  edm::EDGetTokenT<edm::PSimHitContainer> simhitToken_;
  edm::EDGetTokenT<ME0DigiCollection> me0DigiToken_;
  edm::EDGetTokenT<edm::DetSetVector<ME0DigiSimLink> > me0DigiSimLinkToken_;
  bool debug_;

  TH1F *hProces;
  TH1F *hParticleTypes;
  TH1F *hAllSimHitsType;
  TH1F *energy_loss;
  TH1F *processtypeElectrons;
  TH1F *processtypePositrons;
  TH1F *processtypeMuons;
  TH1F *Compton_energy;
  TH1F *EIoni_energy;
  TH1F *PairProd_energy;
  TH1F *Conversions_energy;
  TH1F *EBrem_energy;
  TH1F *Muon_energy;
  TH1F *tof_allPart_bx0;
  TH1F *tof_mu_bx0;
  TH1F *tof_elec_bx0;
};

ME0DigiSimLinkReader::ME0DigiSimLinkReader(const edm::ParameterSet& pset) :
    simhitToken_(consumes < edm::PSimHitContainer > (pset.getParameter < edm::InputTag > ("simhitToken")))
  , me0DigiToken_(consumes < ME0DigiCollection > (pset.getParameter < edm::InputTag > ("me0DigiToken")))
  , me0DigiSimLinkToken_(consumes < edm::DetSetVector<ME0DigiSimLink> > (pset.getParameter < edm::InputTag > ("me0DigiSimLinkToken")))
  , debug_(pset.getParameter<bool>("debugFlag"))
{
  usesResource("TFileService");
  edm::Service < TFileService > fs;
  hProces = fs->make < TH1F > ("hProces", "Process type for all the simHits", 20, 0, 20);
  hAllSimHitsType = fs->make < TH1F > ("hAllSimHitsType", "pdgId for All simHits", 500, 0, 500);
  hParticleTypes = fs->make < TH1F > ("hParticleTypes", "pdgId for digitized simHits", 500, 0, 500);
  energy_loss = fs->make < TH1F > ("energy_loss", "energy_loss", 10000000, 0., 0.001);

  processtypeElectrons = fs->make < TH1F > ("processtypeElectrons", "processtypeElectrons", 20, 0, 20);
  processtypePositrons = fs->make < TH1F > ("processtypePositrons", "processtypePositrons", 20, 0, 20);
  processtypeMuons = fs->make < TH1F > ("processtypeMuons", "processtypeMuons", 20, 0, 20);

  Compton_energy = fs->make < TH1F > ("Compton_energy", "Compton_energy", 10000000, 0., 10);
  EIoni_energy = fs->make < TH1F > ("EIoni_energy", "EIoni_energy", 10000000, 0., 10);
  PairProd_energy = fs->make < TH1F > ("PairProd_energy", "PairProd_energy", 10000000, 0., 10);
  Conversions_energy = fs->make < TH1F > ("Conversions_energy", "Conversions_energy", 10000000, 0., 10);
  EBrem_energy = fs->make < TH1F > ("EBrem_energy", "EBrem_energy", 10000000, 0., 10);
  Muon_energy = fs->make < TH1F > ("Muon_energy", "Muon_energy", 10000000, 0., 1000);

  tof_allPart_bx0 = fs->make < TH1F > ("tof_allPart_bx0", "tof_allPart_bx0", 1000, 0., 100.);
  tof_mu_bx0 = fs->make < TH1F > ("tof_mu_bx0", "tof_mu_bx0", 1000, 0., 100.);
  tof_elec_bx0 = fs->make < TH1F > ("tof_elec_bx0", "tof_elec_bx0", 1000, 0., 100.);

}

void ME0DigiSimLinkReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  edm::Handle < ME0DigiCollection > digis;
  event.getByToken(me0DigiToken_, digis);

  edm::Handle < edm::PSimHitContainer > simHits;
  event.getByToken(simhitToken_, simHits);

  edm::ESHandle < ME0Geometry > pDD;
  eventSetup.get<MuonGeometryRecord>().get(pDD);

  edm::Handle < edm::DetSetVector<ME0DigiSimLink> > theSimlinkDigis;
  event.getByToken(me0DigiSimLinkToken_, theSimlinkDigis);

  //loop over all simhits
  for (const auto& simHit : *simHits)
  {
    hProces->Fill(simHit.processType());
    hAllSimHitsType->Fill(simHit.particleType());

    if (std::abs(simHit.particleType()) == 13){
      Muon_energy->Fill(simHit.pabs());
    }
    else if (std::abs(simHit.particleType()) == 11)
    {
      if (simHit.processType() == 13)
        Compton_energy->Fill(simHit.pabs());
      else if (simHit.processType() == 2)
        EIoni_energy->Fill(simHit.pabs());
      else if (simHit.processType() == 4)
        PairProd_energy->Fill(simHit.pabs());
      else if (simHit.processType() == 14)
        Conversions_energy->Fill(simHit.pabs());
      else if (simHit.processType() == 3)
        EBrem_energy->Fill(simHit.pabs());
    }

  }

//loop over the detectors which have digitized simhits
  for (edm::DetSetVector<ME0DigiSimLink>::const_iterator itsimlink = theSimlinkDigis->begin();
      itsimlink != theSimlinkDigis->end(); itsimlink++)
  {
    int detid = itsimlink->detId();
    if(debug_)
      LogDebug("ME0DigiSimLinkReader") << "detid\t" << detid << std::endl;
    const ME0EtaPartition* roll = pDD->etaPartition(detid);
    const ME0DetId me0Id = roll->id();
    const int nstrips = roll->nstrips();

    double fullAngularStripPitch = 0.;
    Local3DPoint locMuonEntry(0., 0., 0.);
    GlobalPoint globMuonEntry(0., 0., 0.);
    LocalVector lvMu(0., 0., 0.);
    GlobalVector gvMu(0., 0., 0.);

//loop over ME0DigiSimLinks
    for (edm::DetSet<ME0DigiSimLink>::const_iterator link_iter = itsimlink->data.begin();
        link_iter != itsimlink->data.end(); ++link_iter)
    {
      int strip = link_iter->getStrip();
      int processtype = link_iter->getProcessType();
      int particletype = link_iter->getParticleType();
      int bx = link_iter->getBx();
      double partTof = link_iter->getTimeOfFlight();
      double myEnergyLoss = link_iter->getEnergyLoss();
      const StripTopology& topology = roll->specificTopology();
      double angularHalfRoll = (-1) * topology.stripAngle(roll->nstrips());
      double halfAngularStripPitch = angularHalfRoll / nstrips;
      fullAngularStripPitch = 2 * halfAngularStripPitch;

      if (debug_)
      {
        LogDebug("ME0DigiSimLinkReader") << "roll Id\t" << me0Id << std::endl
					 << "number of strips \t" << nstrips << std::endl
					 << "simhit particle type\t" << particletype << std::endl
					 << "simhit process type\t" << processtype << std::endl
					 << "linked to strip with number\t" << strip << std::endl
					 << "in bunch crossing\t" << bx << std::endl
					 << "energy loss\t" << myEnergyLoss << std::endl
					 << "time of flight't" << partTof << std::endl
					 << "roll Id\t" << roll->id() << "\tangularStripCoverage \t" << fullAngularStripPitch << std::endl;
      }

      hParticleTypes->Fill(particletype);

      if (particletype == 11)
        processtypeElectrons->Fill(processtype);
      if (particletype == -11)
        processtypePositrons->Fill(processtype);
      if (std::abs(particletype) == 13)
      {
        processtypeMuons->Fill(processtype);
        locMuonEntry = link_iter->getEntryPoint();
        globMuonEntry = roll->toGlobal(locMuonEntry);
        lvMu = (link_iter->getMomentumAtEntry());
      }

      if (bx == 0)
      {
        if (me0Id.station() != 1)
	  LogDebug("ME0DigiSimLinkReader") << "wrong ME0 station !=1" << std::endl;
        else
        {
          tof_allPart_bx0->Fill(partTof);
          if (std::abs(particletype) == 13)
          {
            tof_mu_bx0->Fill(partTof);
          }
          else if (std::abs(particletype) == 11)
          {
            if (me0Id.station() == 1) tof_elec_bx0->Fill(partTof);
          }
        else
          continue;
        }
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (ME0DigiSimLinkReader);
