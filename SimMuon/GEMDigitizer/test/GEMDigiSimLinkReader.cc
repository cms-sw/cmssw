/** \class GEMDigiSimLinkReader
 *
 *  Dumps GEMDigiSimLinks digis
 *
 *  \authors: Roumyana Hadjiiska
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
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

class GEMDigiSimLinkReader: public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:

  explicit GEMDigiSimLinkReader(const edm::ParameterSet& pset);

  virtual ~GEMDigiSimLinkReader()
  {
  }

  void analyze(const edm::Event &, const edm::EventSetup&);

private:

  edm::EDGetTokenT<edm::PSimHitContainer> simhitToken_;
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken_;
  edm::EDGetTokenT<edm::DetSetVector<GEMDigiSimLink> > gemDigiSimLinkToken_;
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

  TH1F *gemCLS_allClusters;
  TH1F *gemCLS_ClustersWithMuon;
  TH1F *gemCLS_ElectronClusters;
  TH1F *deltaPhi_allStations;
  TH1F *deltaPhi_cls1;
  TH1F *deltaPhi_cls2;
  TH1F *deltaPhi_cls3;
  TH1F *deltaPhi_cls4;
  TH1F *deltaPhi_cls5;

  TH1F *deltaPhi_allStations_normalized;
  TH1F *deltaPhi_GE11;
  TH1F *deltaPhi_GE21long;

  TH2F *mom_cls_allStations;
  TH2F *deltaPhi_cls_allStations;
  TH2F *deltaPhi_cls_allStations_normalized;

  TH2F *mom_cls_GE11;
  TH2F *deltaPhi_cls_GE11;

  TH2F *mom_cls_GE21long;
  TH2F *deltaPhi_cls_GE21long;

  TH1F *allClusters_histo;
  TH1F *muonClusters_histo;

  TH1F *tof_allPart_bx0_ge;
  TH1F *tof_allPart_bx0_ge11;
  TH1F *tof_allPart_bx0_ge21;
  TH1F *tof_mu_bx0_ge11;
  TH1F *tof_mu_bx0_ge21;
  TH1F *tof_elec_bx0_ge11;
  TH1F *tof_elec_bx0_ge21;

};

GEMDigiSimLinkReader::GEMDigiSimLinkReader(const edm::ParameterSet& pset) :
    simhitToken_(consumes < edm::PSimHitContainer > (pset.getParameter < edm::InputTag > ("simhitToken")))
  , gemDigiToken_(consumes < GEMDigiCollection > (pset.getParameter < edm::InputTag > ("gemDigiToken")))
  , gemDigiSimLinkToken_(consumes < edm::DetSetVector<GEMDigiSimLink> > (pset.getParameter < edm::InputTag > ("gemDigiSimLinkToken")))
  , debug_(pset.getParameter<bool>("debugFlag"))
{
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

  gemCLS_allClusters = fs->make < TH1F > ("gemCLS_allClusters", "gemCLS_allClusters", 21, -0.5, 20.5);
  gemCLS_ClustersWithMuon = fs->make < TH1F > ("gemCLS_ClustersWithMuon", "gemCLS_ClustersWithMuon", 21, -0.5, 20.5);
  gemCLS_ElectronClusters = fs->make < TH1F > ("gemCLS_ElectronClusters", "gemCLS_ElectronClusters", 21, -0.5, 20.5);

  deltaPhi_allStations = fs->make < TH1F > ("deltaPhi_allStations", "deltaPhi_allStations", 2000000, -1., 1.);
  deltaPhi_allStations_normalized = fs->make < TH1F > ("deltaPhi_allStations_normalized", "deltaPhi_allStations_normalized", 2000, -10., 10.);
  deltaPhi_GE11 = fs->make < TH1F > ("deltaPhi_GE11", "deltaPhi_GE11", 2000000, -1., 1.);
  deltaPhi_GE21long = fs->make < TH1F > ("deltaPhi_GE21long", "deltaPhi_GE21long", 2000000, -1., 1.);

  mom_cls_allStations = fs->make < TH2F > ("mom_cls_allStations", "mom_cls_allStations", 20000, 0., 2000., 21, -0.5, 20.5);
  mom_cls_allStations->SetXTitle("Muon Momentum [GeV]");
  mom_cls_allStations->SetYTitle("Cluster Size");

  mom_cls_GE11 = fs->make < TH2F > ("mom_cls_GE11", "mom_cls_GE11", 20000, 0., 2000., 21, -0.5, 20.5);
  mom_cls_GE11->SetXTitle("Muon Momentum [GeV]");
  mom_cls_GE11->SetYTitle("Cluster Size");

  mom_cls_GE21long = fs->make < TH2F > ("mom_cls_GE21long", "mom_cls_GE21long", 20000, 0., 2000., 21, -0.5, 20.5);
  mom_cls_GE21long->SetXTitle("Muon Momentum [GeV]");
  mom_cls_GE21long->SetYTitle("Cluster Size");

  deltaPhi_cls_allStations = fs->make < TH2F> ("deltaPhi_cls_allStations", "deltaPhi_cls_allStations", 2000000, -1., 1., 21, -0.5, 20.5);
  deltaPhi_cls_allStations->SetXTitle("Delta #phi [rad]");
  deltaPhi_cls_allStations->SetYTitle("Cluster Size");

  deltaPhi_cls1 = fs->make < TH1F > ("#Delta#phi_cls1", "#Delta#phi for CLS=1", 1000000, -0.5, 0.5);
  deltaPhi_cls1->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls2 = fs->make < TH1F > ("#Delta#phi_cls2", "#Delta#phi for CLS=2", 1000000, -0.5, 0.5);
  deltaPhi_cls2->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls3 = fs->make < TH1F > ("#Delta#phi_cls3", "#Delta#phi for CLS=3", 1000000, -0.5, 0.5);
  deltaPhi_cls3->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls4 = fs->make < TH1F > ("#Delta#phi_cls4", "#Delta#phi for CLS=4", 1000000, -0.5, 0.5);
  deltaPhi_cls4->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls5 = fs->make < TH1F > ("#Delta#phi_cls5", "#Delta#phi for CLS=5", 1000000, -0.5, 0.5);
  deltaPhi_cls5->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls_allStations_normalized = fs->make < TH2F > ("deltaPhi_cls_allStations_normalized", "deltaPhi_cls_allStations_normalized", 2000, -10., 10., 21, -0.5, 20.5);
  deltaPhi_cls_allStations_normalized->SetXTitle("Delta #phi / strip pitch");
  deltaPhi_cls_allStations_normalized->SetYTitle("Cluster Size");

  deltaPhi_cls_GE11 = fs->make < TH2F > ("deltaPhi_cls_GE11", "deltaPhi_cls_GE11", 2000000, -1., 1., 21, -0.5, 20.5);
  deltaPhi_cls_GE11->SetXTitle("Delta #phi [rad]");
  deltaPhi_cls_GE11->SetYTitle("Cluster Size");

  deltaPhi_cls_GE21long = fs->make < TH2F > ("deltaPhi_cls_GE21long", "deltaPhi_cls_GE21long", 2000000, -1., 1., 21, -0.5, 20.5);
  deltaPhi_cls_GE21long->SetXTitle("Delta #phi [rad]");
  deltaPhi_cls_GE21long->SetYTitle("Cluster Size");

  allClusters_histo = fs->make < TH1F > ("allClusters_histo", "allClusters_histo", 51, -0.5, 50.5);
  muonClusters_histo = fs->make < TH1F > ("muonClusters_histo", "muonClusters_histo", 51, -0.5, 50.5);

  tof_allPart_bx0_ge = fs->make < TH1F > ("tof_allPart_bx0_ge", "tof_allPart_bx0_ge", 1000, 0., 100.);
  tof_allPart_bx0_ge11 = fs->make < TH1F > ("tof_allPart_bx0_ge11", "tof_allPart_bx0_ge11", 1000, 0., 100.);
  tof_allPart_bx0_ge21 = fs->make < TH1F > ("tof_allPart_bx0_ge21", "tof_allPart_bx0_ge21", 1000, 0., 100.);
  tof_mu_bx0_ge11 = fs->make < TH1F > ("tof_mu_bx0_ge11", "tof_mu_bx0_ge11", 1000, 0., 100.);
  tof_mu_bx0_ge21 = fs->make < TH1F > ("tof_mu_bx0_ge21", "tof_mu_bx0_ge21", 1000, 0., 100.);
  tof_elec_bx0_ge11 = fs->make < TH1F > ("tof_elec_bx0_ge11", "tof_elec_bx0_ge11", 1000, 0., 100.);
  tof_elec_bx0_ge21 = fs->make < TH1F > ("tof_elec_bx0_ge21", "tof_elec_bx0_ge21", 1000, 0., 100.);

}

void GEMDigiSimLinkReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
  edm::Handle < GEMDigiCollection > digis;
  event.getByToken(gemDigiToken_, digis);

  edm::Handle < edm::PSimHitContainer > simHits;
  event.getByToken(simhitToken_, simHits);

  edm::ESHandle < GEMGeometry > pDD;
  eventSetup.get<MuonGeometryRecord>().get(pDD);

  edm::Handle < edm::DetSetVector<GEMDigiSimLink> > theSimlinkDigis;
  event.getByToken(gemDigiSimLinkToken_, theSimlinkDigis);

  //loop over all simhits
  for (const auto& simHit : *simHits)
  {

    std::cout << "particle type\t" << simHit.particleType() << "\tprocess type\t" << simHit.processType() << std::endl;

    hProces->Fill(simHit.processType());
    hAllSimHitsType->Fill(simHit.particleType());

    if (abs(simHit.particleType()) == 13)
      Muon_energy->Fill(simHit.pabs());
    if (abs(simHit.particleType()) == 11)
    {
      if (simHit.processType() == 13)
        Compton_energy->Fill(simHit.pabs());
      if (simHit.processType() == 2)
        EIoni_energy->Fill(simHit.pabs());
      if (simHit.processType() == 4)
        PairProd_energy->Fill(simHit.pabs());
      if (simHit.processType() == 14)
        Conversions_energy->Fill(simHit.pabs());
      if (simHit.processType() == 3)
        EBrem_energy->Fill(simHit.pabs());
    }

  }

//loop over the detectors which have digitized simhits
  for (edm::DetSetVector<GEMDigiSimLink>::const_iterator itsimlink = theSimlinkDigis->begin();
      itsimlink != theSimlinkDigis->end(); itsimlink++)
  {
    //get the particular detector
    int detid = itsimlink->detId();
    if(debug_)
      std::cout << "detid\t" << detid << std::endl;
    const GEMEtaPartition* roll = pDD->etaPartition(detid);
    const GEMDetId gemId = roll->id();
    const int nstrips = roll->nstrips();

    double fullAngularStripPitch = 0.;
    std::map<int, int> myCluster; //<strip, pdgId>
    std::vector<int> muonFired;
    std::map<int, int> MuCluster; //<strip, pdgId>
    std::map<int, int> ElecCluster; //<strip, pdgId>
    Local3DPoint locMuonEntry(0., 0., 0.);
    GlobalPoint globMuonEntry(0., 0., 0.);
    LocalVector lvMu(0., 0., 0.);
    GlobalVector gvMu(0., 0., 0.);
    double muMomentum = 0.;

    double simMuPhi = 0.;
    double deltaPhi = 0.;

//loop over GemDigiSimLinks
    for (edm::DetSet<GEMDigiSimLink>::const_iterator link_iter = itsimlink->data.begin();
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
        std::cout << "roll Id\t" << gemId << std::endl;
        std::cout << "number of strips \t" << nstrips << std::endl;
        std::cout << "simhit particle type\t" << particletype << std::endl;
        std::cout << "simhit process type\t" << processtype << std::endl;
        std::cout << "linked to strip with number\t" << strip << std::endl;
        std::cout << "in bunch crossing\t" << bx << std::endl;
        std::cout << "energy loss\t" << myEnergyLoss << std::endl;
        std::cout << "time of flight't" << partTof << std::endl;
        std::cout << "roll Id\t" << roll->id() << "\tangularStripCoverage \t" << fullAngularStripPitch << std::endl;
      }

      hParticleTypes->Fill(particletype);

      if (particletype == 11)
        processtypeElectrons->Fill(processtype);
      if (particletype == -11)
        processtypePositrons->Fill(processtype);
      if (abs(particletype) == 13)
      {
        std::cout << "particle\t" << particletype << "\tdetektor\t" << gemId << std::endl;
        processtypeMuons->Fill(processtype);
        locMuonEntry = link_iter->getEntryPoint();
        globMuonEntry = roll->toGlobal(locMuonEntry);
        simMuPhi = globMuonEntry.phi();
        lvMu = (link_iter->getMomentumAtEntry());
        muMomentum = gvMu.mag();
      }

      if (bx == 0)
      {
        tof_allPart_bx0_ge->Fill(partTof);
        if (gemId.station() == 1) tof_allPart_bx0_ge11->Fill(partTof);
        if (gemId.station() == 3) tof_allPart_bx0_ge21->Fill(partTof);
        if (abs(particletype) == 13)
          {
            if (gemId.station() == 1) tof_mu_bx0_ge11->Fill(partTof);
            if (gemId.station() == 3) tof_mu_bx0_ge21->Fill(partTof);
            MuCluster.insert(std::pair<int, int>(strip, particletype));
          }
        else if (abs(particletype) == 11)
        {
          if (gemId.station() == 1) tof_elec_bx0_ge11->Fill(partTof);
          if (gemId.station() == 3) tof_elec_bx0_ge21->Fill(partTof);
          ElecCluster.insert(std::pair<int, int>(strip, particletype));
        }
        else
          continue;
      }
    }

    if (MuCluster.size() != 0)
    {
      for (std::map<int, int>::iterator it = MuCluster.begin(); it != MuCluster.end(); ++it)
      {
        myCluster.insert(std::pair<int, int>(it->first, it->second));
      }
    }

    if (MuCluster.size() != 0)
    {
      for (std::map<int, int>::iterator it = MuCluster.begin(); it != MuCluster.end(); ++it)
      {
        myCluster.insert(std::pair<int, int>(it->first, it->second));
      }
    }

    if (ElecCluster.size() != 0)
    {
      for (std::map<int, int>::iterator it = ElecCluster.begin(); it != ElecCluster.end(); ++it)
      {
        myCluster.insert(std::pair<int, int>(it->first, it->second));
      }
    }

    if (MuCluster.size() != 0)
    {
      for (std::map<int, int>::iterator it = myCluster.begin(); it != myCluster.end(); ++it)
      {
        if (abs(it->second) == 13)
        {
          muonFired.push_back(it->first);
        }
      }
    }

    if (myCluster.size() != 0)
    {
      std::cout << "=+=+=+=+=+=+=+=" << std::endl;
      std::cout << "Muon size " << muonFired.size() << std::endl;

      std::vector<int> allFired;
      std::vector<std::vector<int> > tempCluster;
      for (std::map<int, int>::iterator it = myCluster.begin(); it != myCluster.end(); ++it)
      {
        allFired.push_back(it->first);
      }

      int clusterInd = 0;
      for (unsigned int kk = 0; kk < allFired.size(); kk++)
      {
        std::cout << "kk\t" << kk << std::endl;
        int myDelta = 0;
        std::vector<int> prazen;
        tempCluster.push_back(prazen);
        (tempCluster[clusterInd]).push_back(allFired[kk]);
        unsigned int i = kk;
        std::cout << "i\t" << i << "\tpush kk\t" << allFired[kk] << "\tclusterInd\t" << clusterInd << std::endl;
        for (; i < allFired.size(); i++)
        {
          if (i + 1 < allFired.size())
          {
            myDelta = allFired[i + 1] - allFired[i];
            if (myDelta == 1)
            {
              tempCluster[clusterInd].push_back(allFired[i + 1]);
              std::cout << "i\t" << i << "\ti+1\t" << i + 1 << "\tpush i+1\t" << allFired[i + 1] << std::endl;
            }
            else
              break;
          }
        }
        kk = i + 2;
        clusterInd++;
      }

      int firstStrip = 0;
      int lastStrip = 0;
      int muonCluster = 0;
      GlobalPoint pointDigiHit;
      allClusters_histo->Fill(tempCluster.size());

      for (unsigned int j = 0; j < tempCluster.size(); j++)
      {
        bool checkMu = false;
        unsigned int tempSize = (tempCluster[j]).size();
        gemCLS_allClusters->Fill((tempCluster[j]).size());
        for (unsigned int l = 0; l < (tempCluster[j]).size(); ++l)
        {
          std::vector<int>::iterator muIt = find(muonFired.begin(), muonFired.end(), (tempCluster[j])[l]);
          if (muIt != muonFired.end())
          {
            checkMu = true;
          }
          else
          {
            checkMu = false;
          }
          if (checkMu)
            muonCluster++;
        }

        firstStrip = (tempCluster[j])[0];
        lastStrip = (tempCluster[j])[tempSize - 1];

        if (firstStrip == lastStrip)
          pointDigiHit = roll->toGlobal(roll->centreOfStrip(firstStrip));
        else
        {
          double myDeltaX = (roll->centreOfStrip(lastStrip).x() + roll->centreOfStrip(firstStrip).x()) / 2.;
          double myDeltaY = (roll->centreOfStrip(lastStrip).y() + roll->centreOfStrip(firstStrip).y()) / 2.;
          double myDeltaZ = (roll->centreOfStrip(lastStrip).y() + roll->centreOfStrip(firstStrip).z()) / 2.;
          Local3DPoint locDigi(myDeltaX, myDeltaY, myDeltaZ);
          pointDigiHit = roll->toGlobal(locDigi);
        }

        double digiPhi = pointDigiHit.phi();
        if (checkMu)
        {
          gemCLS_ClustersWithMuon->Fill((tempCluster[j]).size());
          deltaPhi = simMuPhi - digiPhi;
          deltaPhi_allStations->Fill(deltaPhi);
          if ((tempCluster[j]).size() == 1)
            deltaPhi_cls1->Fill(deltaPhi);
          if ((tempCluster[j]).size() == 2)
            deltaPhi_cls2->Fill(deltaPhi);
          if ((tempCluster[j]).size() == 3)
            deltaPhi_cls3->Fill(deltaPhi);
          if ((tempCluster[j]).size() == 4)
            deltaPhi_cls4->Fill(deltaPhi);
          if ((tempCluster[j]).size() == 5)
            deltaPhi_cls5->Fill(deltaPhi);

          deltaPhi_allStations_normalized->Fill(deltaPhi / fullAngularStripPitch);
          mom_cls_allStations->Fill(muMomentum, (tempCluster[j]).size());
          deltaPhi_cls_allStations->Fill(deltaPhi, (tempCluster[j]).size());
          deltaPhi_cls_allStations_normalized->Fill(deltaPhi / fullAngularStripPitch, (tempCluster[j]).size());
          if (gemId.station() == 1)
          {
            deltaPhi_GE11->Fill(deltaPhi);
            mom_cls_GE11->Fill(muMomentum, (tempCluster[j]).size());
            deltaPhi_cls_GE11->Fill(deltaPhi, (tempCluster[j]).size());
          }
          if (gemId.station() == 3)
          {
            deltaPhi_GE21long->Fill(deltaPhi);
            mom_cls_GE21long->Fill(muMomentum, (tempCluster[j]).size());
            deltaPhi_cls_GE21long->Fill(deltaPhi, (tempCluster[j]).size());
          }
        }
        else
          gemCLS_ElectronClusters->Fill((tempCluster[j]).size());
      }        //end tempCluster

      muonClusters_histo->Fill(muonCluster);

    }        //end myCluster!=0

  }        //end given detector
}

#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE (GEMDigiSimLinkReader);

