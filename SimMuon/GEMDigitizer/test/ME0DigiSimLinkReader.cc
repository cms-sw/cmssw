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
#include "TH2F.h"

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

  TH1F *me0CLS_allClusters;
  TH1F *me0CLS_ClustersWithMuon;
  TH1F *me0CLS_ElectronClusters;

  TH1F *hdeltaPhi;
  TH1F *deltaPhi_normalized;

  TH2F *deltaPhi_cls;
  TH2F *deltaPhi_cls_normalized;
  TH2F *mom_cls;

  TH1F *deltaPhi_cls1;
  TH1F *deltaPhi_cls2;
  TH1F *deltaPhi_cls3;
  TH1F *deltaPhi_cls4;
  TH1F *deltaPhi_cls5;

  TH1F *allClusters_histo;
  TH1F *muonClusters_histo;

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

  me0CLS_allClusters = fs->make < TH1F > ("me0CLS_allClusters", "me0CLS_allClusters", 21, -0.5, 20.5);
  me0CLS_ClustersWithMuon = fs->make < TH1F > ("me0CLS_ClustersWithMuon", "me0CLS_ClustersWithMuon", 21, -0.5, 20.5);
  me0CLS_ElectronClusters = fs->make < TH1F > ("me0CLS_ElectronClusters", "me0CLS_ElectronClusters", 21, -0.5, 20.5);

  hdeltaPhi = fs->make < TH1F > ("hdeltaPhi", "hdeltaPhi", 2000000, -1., 1.);
  deltaPhi_normalized = fs->make < TH1F > ("deltaPhi_normalized", "deltaPhi_normalized", 2000, -10., 10.);

  deltaPhi_cls = fs->make < TH2F > ("deltaPhi_cls", "deltaPhi_cls", 2000000, -1., 1., 21, -0.5, 20.5);
  deltaPhi_cls->SetXTitle("Delta #phi [rad]");
  deltaPhi_cls->SetYTitle("Cluster Size");

  deltaPhi_cls_normalized = fs->make < TH2F > ("deltaPhi_cls_normalized", "deltaPhi_cls_normalized", 2000, -10., 10., 21, -0.5, 20.5);
  deltaPhi_cls_normalized->SetXTitle("Delta #phi / strip pitch");
  deltaPhi_cls_normalized->SetYTitle("Cluster Size");

  mom_cls = fs->make < TH2F > ("mom_cls", "mom_cls", 20000, 0., 2000., 21, -0.5, 20.5);
  mom_cls->SetXTitle("Muon Momentum [GeV]");
  mom_cls->SetYTitle("Cluster Size");

  deltaPhi_cls1 = fs->make < TH1F > ("#Delta#phi_cls1", "#Delta#phi for CLS=1", 1000000, -0.5, 0.5);
  deltaPhi_cls1->SetXTitle("#Delta#phi [rad]");
  deltaPhi_cls->SetYTitle("Cluster Size");

  deltaPhi_cls2 = fs->make < TH1F > ("#Delta#phi_cls2", "#Delta#phi for CLS=2", 1000000, -0.5, 0.5);
  deltaPhi_cls2->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls3 = fs->make < TH1F > ("#Delta#phi_cls3", "#Delta#phi for CLS=3", 1000000, -0.5, 0.5);
  deltaPhi_cls3->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls4 = fs->make < TH1F > ("#Delta#phi_cls4", "#Delta#phi for CLS=4", 1000000, -0.5, 0.5);
  deltaPhi_cls4->SetXTitle("#Delta#phi [rad]");

  deltaPhi_cls5 = fs->make < TH1F > ("#Delta#phi_cls5", "#Delta#phi for CLS=5", 1000000, -0.5, 0.5);
  deltaPhi_cls5->SetXTitle("#Delta#phi [rad]");

  allClusters_histo = fs->make < TH1F > ("allClusters_histo", "allClusters_histo", 51, -0.5, 50.5);
  muonClusters_histo = fs->make < TH1F > ("muonClusters_histo", "muonClusters_histo", 51, -0.5, 50.5);

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
  for (edm::DetSetVector<ME0DigiSimLink>::const_iterator itsimlink = theSimlinkDigis->begin();
      itsimlink != theSimlinkDigis->end(); itsimlink++)
  {
    //get the particular detector
    int detid = itsimlink->detId();
    if(debug_)
      std::cout << "detid\t" << detid << std::endl;
    const ME0EtaPartition* roll = pDD->etaPartition(detid);
    const ME0DetId me0Id = roll->id();
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
        std::cout << "roll Id\t" << me0Id << std::endl;
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
        std::cout << "particle\t" << particletype << "\tdetektor\t" << me0Id << std::endl;
        processtypeMuons->Fill(processtype);
        locMuonEntry = link_iter->getEntryPoint();
        globMuonEntry = roll->toGlobal(locMuonEntry);
        simMuPhi = globMuonEntry.phi();
        lvMu = (link_iter->getMomentumAtEntry());
        muMomentum = gvMu.mag();
      }

      if (bx == 0)
      {
        if (me0Id.station() != 1) std::cout << "wrong ME0 station !=1" << std::endl;
        else
        {
          tof_allPart_bx0->Fill(partTof);
          if (abs(particletype) == 13)
          {
            tof_mu_bx0->Fill(partTof);
            MuCluster.insert(std::pair<int, int>(strip, particletype));
          }
          else if (abs(particletype) == 11)
          {
            if (me0Id.station() == 1) tof_elec_bx0->Fill(partTof);
            ElecCluster.insert(std::pair<int, int>(strip, particletype));
          }
        else
          continue;
        }
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
        me0CLS_allClusters->Fill((tempCluster[j]).size());
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
          if (me0Id.station() != 1) std::cout << "wrong ME0 station !=1" << std::endl;
          else
          {
            me0CLS_ClustersWithMuon->Fill((tempCluster[j]).size());
            deltaPhi = simMuPhi - digiPhi;

            hdeltaPhi->Fill(deltaPhi);
            deltaPhi_normalized->Fill(deltaPhi / fullAngularStripPitch);

            deltaPhi_cls->Fill(deltaPhi, (tempCluster[j]).size());
            deltaPhi_cls_normalized->Fill(deltaPhi / fullAngularStripPitch, (tempCluster[j]).size());
            mom_cls->Fill(muMomentum, (tempCluster[j]).size());

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
          }
        }
        else
          me0CLS_ElectronClusters->Fill((tempCluster[j]).size());
      }        //end tempCluster

      muonClusters_histo->Fill(muonCluster);

    }        //end myCluster!=0

  }        //end given detector
}

#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE (ME0DigiSimLinkReader);

