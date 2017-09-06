// user include files

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include <map>
#include <array>
#include <string>
#include <numeric>

class HGCalShowerSeparation : public DQMEDAnalyzer {
 public:
  explicit HGCalShowerSeparation(const edm::ParameterSet&);
  ~HGCalShowerSeparation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void fillWithRecHits(std::map<DetId, const HGCRecHit*>&, DetId, unsigned int,
                       float, int&, float&);

  edm::EDGetTokenT<HGCRecHitCollection> recHitsEE_;
  edm::EDGetTokenT<HGCRecHitCollection> recHitsFH_;
  edm::EDGetTokenT<HGCRecHitCollection> recHitsBH_;
  edm::EDGetTokenT<std::vector<CaloParticle> > caloParticles_;

  int debug_;
  bool filterOnEnergyAndCaloP_;
  hgcal::RecHitTools recHitTools_;

  MonitorElement* eta1_;
  MonitorElement* eta2_;
  MonitorElement* energy1_;
  MonitorElement* energy2_;
  MonitorElement* energytot_;
  MonitorElement* scEnergy_;
  MonitorElement* showerProfile_;
  MonitorElement* layerEnergy_;
  MonitorElement* layerDistance_;
  MonitorElement* etaPhi_;
  MonitorElement* deltaEtaPhi_;
  std::vector<MonitorElement*> profileOnLayer_;
  std::vector<MonitorElement*> globalProfileOnLayer_;
  std::vector<MonitorElement*> distanceOnLayer_;
  std::vector<MonitorElement*> idealDistanceOnLayer_;
  std::vector<MonitorElement*> idealDeltaXY_;
  std::vector<MonitorElement*> centers_;

  static const int layers_ = 52;
};

HGCalShowerSeparation::HGCalShowerSeparation(const edm::ParameterSet& iConfig)
  : debug_(iConfig.getParameter<int>("debug")),
    filterOnEnergyAndCaloP_(iConfig.getParameter<bool>("filterOnEnergyAndCaloP")){
  auto recHitsEE = iConfig.getParameter<edm::InputTag>("recHitsEE");
  auto recHitsFH = iConfig.getParameter<edm::InputTag>("recHitsFH");
  auto recHitsBH = iConfig.getParameter<edm::InputTag>("recHitsBH");
  auto caloParticles = iConfig.getParameter<edm::InputTag>("caloParticles");
  recHitsEE_ = consumes<HGCRecHitCollection>(recHitsEE);
  recHitsFH_ = consumes<HGCRecHitCollection>(recHitsFH);
  recHitsBH_ = consumes<HGCRecHitCollection>(recHitsBH);
  caloParticles_ = consumes<std::vector<CaloParticle> >(caloParticles);
}

HGCalShowerSeparation::~HGCalShowerSeparation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HGCalShowerSeparation::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& iRun,
                                         edm::EventSetup const& /* iSetup */) {
  ibooker.cd();
  ibooker.setCurrentFolder("HGCalShowerSeparation");
  scEnergy_ = ibooker.book1D("SCEnergy", "SCEnergy", 240, 0., 120.);
  eta1_ = ibooker.book1D("eta1", "eta1", 80, 0., 4.);
  eta2_ = ibooker.book1D("eta2", "eta2", 80, 0., 4.);
  energy1_ = ibooker.book1D("energy1", "energy1", 240, 0., 120.);
  energy2_ = ibooker.book1D("energy2", "energy2", 240, 0., 120.);
  energytot_ = ibooker.book1D("energytot", "energytot", 200, 100., 200.);
  showerProfile_ = ibooker.book2D("ShowerProfile", "ShowerProfile",
				  800, -400., 400.,
				  layers_, 0., (float)layers_);
  layerEnergy_ =  ibooker.book2D("LayerEnergy", "LayerEnergy",
				  60, 0., 60.,
				  50, 0., 0.1);
  layerDistance_ =  ibooker.book2D("LayerDistance", "LayerDistance",
				   60, 0., 60.,
				   400, -400., 400.);
  etaPhi_ =  ibooker.book2D("EtaPhi", "EtaPhi",
				  800, -4., 4.,
				  800, -4., 4.);
  deltaEtaPhi_ =  ibooker.book2D("DeltaEtaPhi", "DeltaEtaPhi",
				  100, -0.5, 0.5,
				  100, -0.5, 0.5);
  for (int i = 0; i < layers_; ++i) {
    profileOnLayer_.push_back(ibooker.book2D(std::string("ProfileOnLayer_") + std::to_string(i),
					     std::string("ProfileOnLayer_") + std::to_string(i),
					     120, -600., 600.,
					     120, -600., 600.)
			      );
    globalProfileOnLayer_.push_back(ibooker.book2D(std::string("GlobalProfileOnLayer_") + std::to_string(i),
					     std::string("GlobalProfileOnLayer_") + std::to_string(i),
					     320, -160., 160.,
					     320, -160., 160.)
			      );
    distanceOnLayer_.push_back(ibooker.book1D(std::string("DistanceOnLayer_") + std::to_string(i),
					      std::string("DistanceOnLayer_") + std::to_string(i),
					      120, -600., 600.)
			      );
    idealDistanceOnLayer_.push_back(ibooker.book1D(std::string("IdealDistanceOnLayer_") + std::to_string(i),
						   std::string("IdealDistanceOnLayer_") + std::to_string(i),
						   120, -600., 600.)
				    );
    idealDeltaXY_.push_back(ibooker.book2D(std::string("IdealDeltaXY_") + std::to_string(i),
					   std::string("IdealDeltaXY_") + std::to_string(i),
					   800, -400., 400.,
					   800, -400., 400.)
				    );
    centers_.push_back(ibooker.book2D(std::string("Centers_") + std::to_string(i),
				      std::string("Centers_") + std::to_string(i),
				      320, -1600., 1600.,
				      320, -1600., 1600.)
		       );
 }
}

void HGCalShowerSeparation::analyze(const edm::Event& iEvent,
                                  const edm::EventSetup& iSetup) {
  using namespace edm;

  recHitTools_.getEventSetup(iSetup);

  Handle<HGCRecHitCollection> recHitHandleEE;
  Handle<HGCRecHitCollection> recHitHandleFH;
  Handle<HGCRecHitCollection> recHitHandleBH;

  Handle<std::vector<CaloParticle> > caloParticleHandle;
  iEvent.getByToken(caloParticles_, caloParticleHandle);
  const std::vector<CaloParticle>& caloParticles = *caloParticleHandle;


  iEvent.getByToken(recHitsEE_, recHitHandleEE);
  iEvent.getByToken(recHitsFH_, recHitHandleFH);
  iEvent.getByToken(recHitsBH_, recHitHandleBH);
  const auto& rechitsEE = *recHitHandleEE;
  const auto& rechitsFH = *recHitHandleFH;
  const auto& rechitsBH = *recHitHandleBH;

  std::map<DetId, const HGCRecHit*> hitmap;
  for (unsigned int i = 0; i < rechitsEE.size(); ++i) {
    hitmap[rechitsEE[i].detid()] = &rechitsEE[i];
  }
  for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
    hitmap[rechitsFH[i].detid()] = &rechitsFH[i];
  }
  for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
    hitmap[rechitsBH[i].detid()] = &rechitsBH[i];
  }

  // loop over caloParticles
  IfLogTrace(debug_ > 0, "HGCalShowerSeparation")
    << "Number of caloParticles: " << caloParticles.size() << std::endl;
  if (caloParticles.size() == 2) {
    auto eta1 = caloParticles[0].eta();
    auto phi1 = caloParticles[0].phi();
    auto theta1 = 2.*atan(exp(-eta1));
    auto eta2 = caloParticles[1].eta();
    auto phi2 = caloParticles[1].phi();
    auto theta2 = 2.*atan(exp(-eta2));
    eta1_->Fill(eta1);
    eta2_->Fill(eta2);


    // Select event only if the sum of the energy of its recHits
    // is close enough to the gen energy
    int count = 0;
    int size = 0;
    float energy = 0.;
    float energy_tmp = 0.;
    for (const auto& it_caloPart : caloParticles) {
      count++;
      const SimClusterRefVector& simClusterRefVector = it_caloPart.simClusters();
      size += simClusterRefVector.size();
      for (const auto& it_sc : simClusterRefVector) {
	const SimCluster& simCluster = (*(it_sc));
	const std::vector<std::pair<uint32_t, float> >& hits_and_fractions =
          simCluster.hits_and_fractions();
	for (const auto& it_haf : hits_and_fractions) {
	  if (hitmap.count(it_haf.first))
	    energy += hitmap[it_haf.first]->energy()*it_haf.second;
	} //hits and fractions
      } // simcluster
      if (count == 1) {
	energy1_->Fill(energy);
	energy_tmp = energy;
      } else {
	energy2_->Fill(energy-energy_tmp);
      }
    } // caloParticle
    energytot_->Fill(energy);
    if (filterOnEnergyAndCaloP_ && (energy < 2.*0.8*80 or size !=2))
      return;

    deltaEtaPhi_->Fill(eta1-eta2, phi1-phi2);

    for (const auto& it_caloPart : caloParticles) {
      const SimClusterRefVector& simClusterRefVector = it_caloPart.simClusters();
      IfLogTrace(debug_ > 0, "HGCalShowerSeparation")
	<< ">>> " << simClusterRefVector.size() << std::endl;
      for (const auto& it_sc : simClusterRefVector) {
	const SimCluster& simCluster = (*(it_sc));
	if (simCluster.energy() < 80*0.8)
	  continue;
	scEnergy_->Fill(simCluster.energy());
	IfLogTrace(debug_ > 1, "HGCalShowerSeparation")
	  << ">>> SC.energy(): " << simCluster.energy()
	  << " SC.simEnergy(): " << simCluster.simEnergy()
	  << std::endl;
	const std::vector<std::pair<uint32_t, float> >& hits_and_fractions =
          simCluster.hits_and_fractions();

	for (const auto& it_haf : hits_and_fractions) {
	  if (!hitmap.count(it_haf.first))
	    continue;
	  unsigned int hitlayer = recHitTools_.getLayerWithOffset(it_haf.first);
	  auto global = recHitTools_.getPosition(it_haf.first);
	  float globalx = global.x();
	  float globaly = global.y();
	  float globalz = global.z();
	  if (globalz == 0)
	    continue;
	  auto rho1 = globalz*tan(theta1);
	  auto rho2 = globalz*tan(theta2);
	  auto x1 = rho1*cos(phi1);
	  auto y1 = rho1*sin(phi1);
	  auto x2 = rho2*cos(phi2);
	  auto y2 = rho2*sin(phi2);
	  auto half_point_x = (x1+x2)/2.;
	  auto half_point_y = (y1+y2)/2.;
	  auto half_point = sqrt((x1-half_point_x)*(x1-half_point_x)+(y1-half_point_y)*(y1-half_point_y));
	  auto d_len = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
	  auto dn_x = (x2-x1)/d_len;
	  auto dn_y = (y2-y1)/d_len;
	  auto distance = (globalx-x1)*dn_x + (globaly - y1)*dn_y;
	  distance -= half_point;
	  auto idealDistance = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
	  if (hitmap.count(it_haf.first)) {
	    profileOnLayer_[hitlayer]->Fill(10.*(globalx-half_point_x),
					    10.*(globaly-half_point_y),
					    hitmap[it_haf.first]->energy()*it_haf.second);
	    profileOnLayer_[55]->Fill(10.*(globalx-half_point_x),
				      10.*(globaly-half_point_y),
				      hitmap[it_haf.first]->energy()*it_haf.second);
	    globalProfileOnLayer_[hitlayer]->Fill(globalx,
						  globaly,
						  hitmap[it_haf.first]->energy()*it_haf.second);
	    globalProfileOnLayer_[55]->Fill(globalx,
					    globaly,
					    hitmap[it_haf.first]->energy()*it_haf.second);
	    layerEnergy_->Fill(hitlayer, hitmap[it_haf.first]->energy());
	    layerDistance_->Fill(hitlayer, std::abs(10.*distance), hitmap[it_haf.first]->energy()*it_haf.second);
	    etaPhi_->Fill(global.eta(), global.phi());
	    distanceOnLayer_[hitlayer]->Fill(10.*distance);//,
	    idealDistanceOnLayer_[hitlayer]->Fill(10.*idealDistance);//,
	    idealDeltaXY_[hitlayer]->Fill(10.*(x1-x2), 10.*(y1-y2));//,
	    centers_[hitlayer]->Fill(10.*half_point_x, 10.*half_point_y);//,
	    IfLogTrace(debug_ > 0, "HGCalShowerSeparation")
	      << ">>> " << distance
	      << " " << hitlayer
	      << " " << hitmap[it_haf.first]->energy()*it_haf.second
	      << std::endl;
	    showerProfile_->Fill(10.*distance,
				 hitlayer,
				 hitmap[it_haf.first]->energy()*it_haf.second);
	  }
	}  // end simHit
      }  // end simCluster
    }  // end caloparticle
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void HGCalShowerSeparation::fillDescriptions(
					   edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("debug", 1);
  desc.add<bool>("filterOnEnergyAndCaloP", false);
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("recHitsEE", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("recHitsFH", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("recHitsBH", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  descriptions.add("hgcalShowerSeparation", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalShowerSeparation);
