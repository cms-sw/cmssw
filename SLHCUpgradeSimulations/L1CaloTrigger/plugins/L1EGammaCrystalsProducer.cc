// -*- C++ -*-
//
// Package: L1CaloTrigger
// Class: L1EGammaCrystalsProducer
//
/**\class L1EGammaCrystalsProducer L1EGammaCrystalsProducer.cc SLHCUpgradeSimulations/L1CaloTrigger/plugin/L1EGammaCrystalsProducer.cc

Description: Produces crystal clusters using crystal-level information

Implementation:
[Notes on implementation]
*/
//
// Original Author: Nick Smith, Alexander Savin
// Created: Tue Apr 22 2014
// $Id$
//
//


#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <iostream>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "SimDataFormats/SLHC/interface/L1EGCrystalCluster.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

class L1EGCrystalClusterProducer : public edm::EDProducer {
   public:
      explicit L1EGCrystalClusterProducer(const edm::ParameterSet&);

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      bool cluster_passes_cuts(const l1slhc::L1EGCrystalCluster& cluster) const;

      CaloGeometryHelper geometryHelper;
      double EtminForStore;
      bool debug;
      bool useECalEndcap;
      class SimpleCaloHit
      {
         public:
            EBDetId id;
            GlobalVector position; // As opposed to GlobalPoint, so we can add them (for weighted average)
            float energy=0.;
            bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
            bool isEndcapHit=false; // If using endcap, we won't be using integer crystal indices
            
         // tool functions
            inline float pt() const{return (position.mag2()>0) ? energy*sin(position.theta()) : 0.;};
            inline float deta(SimpleCaloHit& other) const{return position.eta() - other.position.eta();};
            int dieta(SimpleCaloHit& other) const
            {
               if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
               // int indices do not contain zero
               // Logic from EBDetId::distanceEta() without the abs()
               if (id.ieta() * other.id.ieta() > 0)
                  return id.ieta()-other.id.ieta();
               return id.ieta()-other.id.ieta()-1;
            };
            inline float dphi(SimpleCaloHit& other) const{return reco::deltaPhi(position.phi(), other.position.phi());};
            int diphi(SimpleCaloHit& other) const
            {
               if ( isEndcapHit || other.isEndcapHit ) return 9999; // We shouldn't compare integer indices in endcap, the map is not linear
               // Logic from EBDetId::distancePhi() without the abs()
               int PI = 180;
               int result = id.iphi() - other.id.iphi();
               while (result > PI) result -= 2*PI;
               while (result <= -PI) result += 2*PI;
               return result;
            };
            float distanceTo(SimpleCaloHit& other) const
            {
               // Treat position as a point, measure 3D distance
               // This is used for endcap hits, where we don't have a rectangular mapping
               return (position-other.position).mag();
            };
            bool operator==(SimpleCaloHit& other) const
            {
               if ( id == other.id &&
                    position == other.position &&
                    energy == other.energy &&
                    isEndcapHit == other.isEndcapHit
                  ) return true;
                  
               return false;
            };
      };
};

L1EGCrystalClusterProducer::L1EGCrystalClusterProducer(const edm::ParameterSet& iConfig) :
   EtminForStore(iConfig.getParameter<double>("EtminForStore")),
   debug(iConfig.getUntrackedParameter<bool>("debug", false)),
   useECalEndcap(iConfig.getParameter<bool>("useECalEndcap"))
{
   produces<l1slhc::L1EGCrystalClusterCollection>("EGCrystalCluster");
   produces<l1extra::L1EmParticleCollection>("EGammaCrystal");
}

void L1EGCrystalClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if ( geometryHelper.getEcalBarrelGeometry() == nullptr )
   {
      edm::ESHandle<CaloTopology> theCaloTopology;
      iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
      edm::ESHandle<CaloGeometry> pG;
      iSetup.get<CaloGeometryRecord>().get(pG);
      double bField000 = 4.;
      geometryHelper.setupGeometry(*pG);
      geometryHelper.setupTopology(*theCaloTopology);
      geometryHelper.initialize(bField000);
   }
   
   std::vector<SimpleCaloHit> ecalhits;
   std::vector<SimpleCaloHit> hcalhits;
   
   // Retrieve the ecal barrel hits
   // using RecHits (https://cmssdt.cern.ch/SDT/doxygen/CMSSW_6_1_2_SLHC6/doc/html/d8/dc9/classEcalRecHit.html)
   edm::Handle<EcalRecHitCollection> pcalohits;
   iEvent.getByLabel("ecalRecHit","EcalRecHitsEB",pcalohits);
   for(auto& hit : *pcalohits.product())
   {
      if(hit.energy() > 0.2 && !hit.checkFlag(EcalRecHit::kOutOfTime) && !hit.checkFlag(EcalRecHit::kL1SpikeFlag))
      {
         auto cell = geometryHelper.getEcalBarrelGeometry()->getGeometry(hit.id());
         SimpleCaloHit ehit;
         ehit.id = hit.id();
         // So, apparently there are (at least) two competing basic vector classes being tossed around in
         // cmssw, the calorimeter geometry package likes to use "DataFormats/GeometryVector/interface/GlobalPoint.h"
         // while "DataFormats/Math/interface/Point3D.h" also contains a competing definition of GlobalPoint. Oh well...
         ehit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
         ehit.energy = hit.energy();
         ecalhits.push_back(ehit);
      }
   }
   
   if ( useECalEndcap )
   {
      // Retrieve the ecal endcap hits
      // using RecHits (https://cmssdt.cern.ch/SDT/doxygen/CMSSW_6_1_2_SLHC6/doc/html/d8/dc9/classEcalRecHit.html)
      edm::Handle<EcalRecHitCollection> pcalohitsEndcap;
      iEvent.getByLabel("ecalRecHit","EcalRecHitsEE",pcalohitsEndcap);
      for(auto& hit : *pcalohitsEndcap.product())
      {
         if(hit.energy() > 0.2 && !hit.checkFlag(EcalRecHit::kOutOfTime) && !hit.checkFlag(EcalRecHit::kL1SpikeFlag))
         {
            auto cell = geometryHelper.getEcalEndcapGeometry()->getGeometry(hit.id());
            SimpleCaloHit ehit;
            // endcap cell ids won't have any relation to barrel hits
            ehit.id = hit.id();
            ehit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
            ehit.energy = hit.energy();
            ehit.isEndcapHit = true;
            ecalhits.push_back(ehit);
         }
      }
   }

   // Retrive hcal hits
   edm::Handle<HBHERecHitCollection> hbhecoll;
   iEvent.getByLabel("hbheprereco", hbhecoll);
   for (auto& hit : *hbhecoll.product())
   {
      if ( hit.energy() > 0.1 )
      {
         auto cell = geometryHelper.getHcalGeometry()->getGeometry(hit.id());
         SimpleCaloHit hhit;
         hhit.id = hit.id();
         hhit.position = GlobalVector(cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z());
         hhit.energy = hit.energy();
         hcalhits.push_back(hhit);
      }
   }

   // Cluster containters
   std::auto_ptr<l1slhc::L1EGCrystalClusterCollection> trigCrystalClusters (new l1slhc::L1EGCrystalClusterCollection );
   std::auto_ptr<l1extra::L1EmParticleCollection> l1EGammaCrystal( new l1extra::L1EmParticleCollection );
   
   // Clustering algorithm
   while(true)
   {
      // Find highest pt hit (that's not already used)
      SimpleCaloHit centerhit;
      for(const auto& hit : ecalhits)
      {
         if ( !hit.stale && hit.pt() > centerhit.pt() )
         {
            centerhit = hit;
         }
      }
      // If we are less than 1GeV or out of hits (i.e. when centerhit is default constructed) we stop
      if ( centerhit.pt() <= 1. ) break;
      if ( debug ) std::cout << "-------------------------------------" << std::endl;
      if ( debug ) std::cout << "New cluster: center crystal pt = " << centerhit.pt() << std::endl;

      // Experimental parameters, don't want to bother with hardcoding them in data format
      std::map<std::string, float> params;
      
      // Find the energy-weighted average position,
      // calculate isolation parameter,
      // calculate pileup-corrected pt,
      // and quantify likelihood of a brem
      GlobalVector weightedPosition;
      GlobalVector ECalPileUpVector;
      float totalEnergy = 0.;
      float ECalIsolation = 0.;
      float ECalPileUpEnergy = 0.;
      float upperSideLobePt = 0.;
      float lowerSideLobePt = 0.;
      std::vector<float> crystalPt;
      std::map<int, float> phiStrip;
      for(auto& hit : ecalhits)
      {
         if ( !hit.stale &&
               ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3)
                || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 3.5*1.41 ) )) // endcap crystals are 30mm on a side, 3.5*sqrt(2) cm radius should enclose 3x3
         {
            weightedPosition += hit.position*hit.energy;
            totalEnergy += hit.energy;
            hit.stale = true;
            crystalPt.push_back(hit.pt());
            if ( debug && hit == centerhit )
               std::cout << "\x1B[32m"; // green hilight
            if ( debug && hit.isEndcapHit ) std::cout <<
               "\tCrystal pt=" << hit.pt() <<
               ", eta=" << hit.position.eta() <<
               ", phi=" << hit.position.phi() << "\x1B[0m" << std::endl;
            else if ( debug ) std::cout <<
               "\tCrystal (" << hit.dieta(centerhit) << "," << hit.diphi(centerhit) <<
               ") pt=" << hit.pt() <<
               ", eta=" << hit.position.eta() <<
               ", phi=" << hit.position.phi() << "\x1B[0m" << std::endl;
         }

         if ( abs(hit.dieta(centerhit)) == 0 && abs(hit.diphi(centerhit)) <= 7 )
         {
            phiStrip[hit.diphi(centerhit)] = hit.pt();
         }

         // Isolation and pileup must not use hits used in the cluster
         // As for the endcap hits, well, as far as this algorithm is concerned, caveat emptor...
         if ( !(!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && abs(hit.diphi(centerhit)) < 3)
              && !(centerhit.isEndcapHit && hit.distanceTo(centerhit) < 3.5*1.41 ) )
         {
            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 14 && abs(hit.diphi(centerhit)) < 14)
                 || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 42. ))
            {
               ECalIsolation += hit.pt();
               if ( hit.pt() > 1. )
                  params["nIsoCrystals1"]++;
            }
            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) >= 3 && hit.diphi(centerhit) < 8)
                 || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit) >= 0.0173*3 && hit.dphi(centerhit) < 0.0173*8 ))
            {
               upperSideLobePt += hit.pt();
            }
            if ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) > -8 && hit.diphi(centerhit) <= -3)
                 || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit)*-1 >= 0.0173*3 && hit.dphi(centerhit)*-1 < 0.0173*8 ))
            {
               lowerSideLobePt += hit.pt();
            }
            if ( hit.pt() < 5. &&
                 ( (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 7 && abs(hit.diphi(centerhit)) < 57 )
                  || (centerhit.isEndcapHit && hit.distanceTo(centerhit) < 50.) ))
            {
               ECalPileUpEnergy += hit.energy;
               ECalPileUpVector += hit.position;
            }
         }
      }
      params["uncorrectedE"] = totalEnergy;
      params["uncorrectedPt"] = totalEnergy*sin(weightedPosition.theta());

      // phi strip params
      // lambda returns size of contiguous strip, one-hole strip
      auto countStrip = [&phiStrip](float threshold) -> std::pair<float, float>
      {
         int nContiguous = 1;
         int nOneHole = 1;
         bool firstHole = false;
         for(int i=1; i<=7; ++i)
         {
            if ( phiStrip[i] > threshold && !firstHole )
            {
               nContiguous++;
               nOneHole++;
            }
            else if ( phiStrip[i] > threshold )
               nOneHole++;
            else if ( !firstHole )
               firstHole = true;
            else
               break;
         }
         firstHole = false;
         for(int i=-1; i>=-7; --i)
         {
            if ( phiStrip[i] > threshold && !firstHole )
            {
               nContiguous++;
               nOneHole++;
            }
            else if ( phiStrip[i] > threshold )
               nOneHole++;
            else if ( !firstHole )
               firstHole = true;
            else
               break;
         }
         return std::make_pair<float, float>(nContiguous, nOneHole);
      };
      auto zeropair = countStrip(0.);
      params["phiStripContiguous0"] = zeropair.first;
      params["phiStripOneHole0"] = zeropair.second;
      auto threepair = countStrip(0.03*totalEnergy);
      params["phiStripContiguous3p"] = threepair.first;
      params["phiStripOneHole3p"] = threepair.second;

      // Check if sidelobes should be included in sum
      if ( upperSideLobePt/params["uncorrectedPt"] > 0.1 )
      {
         for(auto& hit : ecalhits)
         {
            if ( !hit.stale &&
                 (  (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) >= 3 && hit.diphi(centerhit) < 8)
                   || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit) >= 0.0173*3 && hit.dphi(centerhit) < 0.0173*8 )
                 ) )
            {
               weightedPosition += hit.position*hit.energy;
               totalEnergy += hit.energy;
               hit.stale = true;
               crystalPt.push_back(hit.pt());
            }
         }
      }
      if ( lowerSideLobePt/params["uncorrectedPt"] > 0.1 )
      {
         for(auto& hit : ecalhits)
         {
            if ( !hit.stale &&
                 (  (!centerhit.isEndcapHit && abs(hit.dieta(centerhit)) < 2 && hit.diphi(centerhit) > -8 && hit.diphi(centerhit) <= -3)
                   || (centerhit.isEndcapHit && fabs(hit.deta(centerhit)) < 0.02 && hit.dphi(centerhit)*-1 >= 0.0173*3 && hit.dphi(centerhit)*-1 < 0.0173*8 )
                 ) )
            {
               weightedPosition += hit.position*hit.energy;
               totalEnergy += hit.energy;
               hit.stale = true;
               crystalPt.push_back(hit.pt());
            }
         }
      }
      // no need to rescale weightedPosition if we only use theta
      float correctedTotalPt = totalEnergy*sin(weightedPosition.theta());
      params["avgIsoCrystalE"] = (params["nIsoCrystals1"] > 0.) ? ECalIsolation/params["nIsoCrystals1"] : 0.;
      params["upperSideLobePt"] = upperSideLobePt;
      params["lowerSideLobePt"] = lowerSideLobePt;
      ECalIsolation /= params["uncorrectedPt"];
      float totalPtPUcorr = params["uncorrectedPt"] - ECalPileUpEnergy*sin(ECalPileUpVector.theta())/19.;
      float bremStrength = params["uncorrectedPt"] / correctedTotalPt;

      if ( debug ) std::cout << "Weighted position eta = " << weightedPosition.eta() << ", phi = " << weightedPosition.phi() << std::endl;
      if ( debug ) std::cout << "Uncorrected Total energy = " << params["uncorrectedE"] << ", total pt = " << params["uncorrectedPt"] << std::endl;
      if ( debug ) std::cout << "Total energy = " << totalEnergy << ", total pt = " << correctedTotalPt << std::endl;
      if ( debug ) std::cout << "Isolation: " << ECalIsolation << std::endl;

      // Calculate H/E
      float hcalEnergy = 0.;
      for(const auto& hit : hcalhits)
      {
         if ( fabs(hit.deta(centerhit)) < 0.15 && fabs(hit.dphi(centerhit)) < 0.15 )
         {
            hcalEnergy += hit.energy;
         }
      }
      float hovere = hcalEnergy/params["uncorrectedE"];

      if ( debug ) std::cout << "H/E: " << hovere << std::endl;
      
      // Form a l1slhc::L1EGCrystalCluster
      reco::Candidate::PolarLorentzVector p4(correctedTotalPt, weightedPosition.eta(), weightedPosition.phi(), 0.);
      l1slhc::L1EGCrystalCluster cluster(p4, hovere, ECalIsolation, centerhit.id, totalPtPUcorr, bremStrength);
      // Save pt array
      cluster.SetCrystalPtInfo(crystalPt);
      params["crystalCount"] = crystalPt.size();
      cluster.SetExperimentalParams(params);
      trigCrystalClusters->push_back(cluster);

      // Save clusters with some cuts
      if ( cluster_passes_cuts(cluster) )
      {
         // Optional min. Et cut
         if ( cluster.pt() >= EtminForStore ) {
            l1EGammaCrystal->push_back(l1extra::L1EmParticle(p4, edm::Ref<L1GctEmCandCollection>(), 0));
         }
      }
   }

   iEvent.put(trigCrystalClusters,"EGCrystalCluster");
   iEvent.put(l1EGammaCrystal, "EGammaCrystal" );
}

bool
L1EGCrystalClusterProducer::cluster_passes_cuts(const l1slhc::L1EGCrystalCluster& cluster) const {
   // cuts were optimized before pt correction was implemented (uncorrectedPt is core 3x5 crystals)
   float cut_pt = cluster.GetExperimentalParam("uncorrectedPt");
   if ( fabs(cluster.eta()) > 1.479 )
   {
      if ( cluster.hovere() < 22./cut_pt
           && cluster.isolation() < 64./cut_pt+0.1
           && cluster.GetCrystalPt(4)/(cluster.GetCrystalPt(0)+cluster.GetCrystalPt(1)) < ( (cut_pt < 40) ? 0.18*(1-cut_pt/70.):0.18*3/7. ) )
      {
         return true;
      }
   }
   else
   {
      if ( cluster.hovere() < 14./cut_pt+0.05
           && cluster.isolation() < 40./cut_pt+0.1
           && cluster.GetCrystalPt(4)/(cluster.GetCrystalPt(0)+cluster.GetCrystalPt(1)) < ( (cut_pt < 30) ? 0.18*(1-cut_pt/100.):0.18*0.7 ) )
      {
         return true;
      }
   }
   return false;
}

DEFINE_FWK_MODULE(L1EGCrystalClusterProducer);
