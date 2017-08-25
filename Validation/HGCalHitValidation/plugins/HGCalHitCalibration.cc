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

class HGCalHitCalibration : public DQMEDAnalyzer {
 public:
  explicit HGCalHitCalibration(const edm::ParameterSet&);
  ~HGCalHitCalibration() override;

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
  edm::EDGetTokenT<std::vector<reco::PFCluster> > hgcalMultiClusters_;
  edm::EDGetTokenT<std::vector<reco::GsfElectron> > electrons_;
  edm::EDGetTokenT<std::vector<reco::Photon> > photons_;

  int algo_;
  bool rawRecHits_;
  int debug_;
  hgcal::RecHitTools recHitTools_;

  std::map<int, MonitorElement*> h_EoP_CPene_calib_fraction_;
  std::map<int, MonitorElement*> hgcal_EoP_CPene_calib_fraction_;
  std::map<int, MonitorElement*> hgcal_ele_EoP_CPene_calib_fraction_;
  std::map<int, MonitorElement*> hgcal_photon_EoP_CPene_calib_fraction_;
  MonitorElement* LayerOccupancy_;

  static const int layers_ = 60;
  std::array<float, layers_> Energy_layer_calib_;
  std::array<float, layers_> Energy_layer_calib_fraction_;
};

HGCalHitCalibration::HGCalHitCalibration(const edm::ParameterSet& iConfig)
  : rawRecHits_(iConfig.getParameter<bool>("rawRecHits")),
    debug_(iConfig.getParameter<int>("debug")) {
  auto detector = iConfig.getParameter<std::string>("detector");
  auto recHitsEE = iConfig.getParameter<edm::InputTag>("recHitsEE");
  auto recHitsFH = iConfig.getParameter<edm::InputTag>("recHitsFH");
  auto recHitsBH = iConfig.getParameter<edm::InputTag>("recHitsBH");
  auto caloParticles = iConfig.getParameter<edm::InputTag>("caloParticles");
  auto hgcalMultiClusters = iConfig.getParameter<edm::InputTag>("hgcalMultiClusters");
  auto electrons = iConfig.getParameter<edm::InputTag>("electrons");
  auto photons = iConfig.getParameter<edm::InputTag>("photons");
  if (detector == "all") {
    recHitsEE_ = consumes<HGCRecHitCollection>(recHitsEE);
    recHitsFH_ = consumes<HGCRecHitCollection>(recHitsFH);
    recHitsBH_ = consumes<HGCRecHitCollection>(recHitsBH);
    algo_ = 1;
  } else if (detector == "EM") {
    recHitsEE_ = consumes<HGCRecHitCollection>(recHitsEE);
    algo_ = 2;
  } else if (detector == "HAD") {
    recHitsFH_ = consumes<HGCRecHitCollection>(recHitsFH);
    recHitsBH_ = consumes<HGCRecHitCollection>(recHitsBH);
    algo_ = 3;
  }
  caloParticles_ = consumes<std::vector<CaloParticle> >(caloParticles);
  hgcalMultiClusters_ = consumes<std::vector<reco::PFCluster> >(hgcalMultiClusters);
  electrons_ = consumes<std::vector<reco::GsfElectron> >(electrons);
  photons_ = consumes<std::vector<reco::Photon> >(photons);

  // size should be HGC layers 52 is enough
  Energy_layer_calib_.fill(0.);
  Energy_layer_calib_fraction_.fill(0.);
}

HGCalHitCalibration::~HGCalHitCalibration() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void HGCalHitCalibration::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& iRun,
                                         edm::EventSetup const& /* iSetup */) {
  ibooker.cd();
  ibooker.setCurrentFolder("HGCalHitCalibration");
  h_EoP_CPene_calib_fraction_[100] =
      ibooker.book1D("h_EoP_CPene_100_calib_fraction", "", 1000, -0.5, 2.5);
  h_EoP_CPene_calib_fraction_[200] =
      ibooker.book1D("h_EoP_CPene_200_calib_fraction", "", 1000, -0.5, 2.5);
  h_EoP_CPene_calib_fraction_[300] =
      ibooker.book1D("h_EoP_CPene_300_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_EoP_CPene_calib_fraction_[100] =
      ibooker.book1D("hgcal_EoP_CPene_100_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_EoP_CPene_calib_fraction_[200] =
      ibooker.book1D("hgcal_EoP_CPene_200_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_EoP_CPene_calib_fraction_[300] =
      ibooker.book1D("hgcal_EoP_CPene_300_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_ele_EoP_CPene_calib_fraction_[100] =
      ibooker.book1D("hgcal_ele_EoP_CPene_100_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_ele_EoP_CPene_calib_fraction_[200] =
      ibooker.book1D("hgcal_ele_EoP_CPene_200_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_ele_EoP_CPene_calib_fraction_[300] =
      ibooker.book1D("hgcal_ele_EoP_CPene_300_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_photon_EoP_CPene_calib_fraction_[100] =
      ibooker.book1D("hgcal_photon_EoP_CPene_100_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_photon_EoP_CPene_calib_fraction_[200] =
      ibooker.book1D("hgcal_photon_EoP_CPene_200_calib_fraction", "", 1000, -0.5, 2.5);
  hgcal_photon_EoP_CPene_calib_fraction_[300] =
      ibooker.book1D("hgcal_photon_EoP_CPene_300_calib_fraction", "", 1000, -0.5, 2.5);
  LayerOccupancy_ = ibooker.book1D("LayerOccupancy", "", layers_, 0., (float)layers_);
}

void HGCalHitCalibration::fillWithRecHits(
    std::map<DetId, const HGCRecHit*>& hitmap, const DetId hitid,
    const unsigned int hitlayer, const float fraction, int& seedDet,
    float& seedEnergy) {
  if (hitmap.find(hitid) == hitmap.end()) {
    // Hit was not reconstructed
    IfLogTrace(debug_ > 0, "HGCalHitCalibration")
      << ">>> Failed to find detid " << hitid.rawId()
      << " Det " << hitid.det()
      << " Subdet " << hitid.subdetId() << std::endl;
    return;
  }
  unsigned int layer = recHitTools_.getLayerWithOffset(hitid);
  assert(hitlayer == layer);
  Energy_layer_calib_fraction_[layer] += hitmap[hitid]->energy() * fraction;
  LayerOccupancy_->Fill(layer);
  if (seedEnergy < hitmap[hitid]->energy()) {
    seedEnergy = hitmap[hitid]->energy();
    seedDet = recHitTools_.getSiThickness(hitid);
  }
}

void HGCalHitCalibration::analyze(const edm::Event& iEvent,
                                  const edm::EventSetup& iSetup) {
  using namespace edm;

  recHitTools_.getEventSetup(iSetup);

  Handle<HGCRecHitCollection> recHitHandleEE;
  Handle<HGCRecHitCollection> recHitHandleFH;
  Handle<HGCRecHitCollection> recHitHandleBH;

  Handle<std::vector<CaloParticle> > caloParticleHandle;
  iEvent.getByToken(caloParticles_, caloParticleHandle);
  const std::vector<CaloParticle>& caloParticles = *caloParticleHandle;

  Handle<std::vector<reco::PFCluster> > hgcalMultiClustersHandle;
    iEvent.getByToken(hgcalMultiClusters_, hgcalMultiClustersHandle);

  Handle<std::vector<reco::GsfElectron> > PFElectronHandle;
  iEvent.getByToken(electrons_, PFElectronHandle);

  Handle<std::vector<reco::Photon> > PFPhotonHandle;
  iEvent.getByToken(photons_, PFPhotonHandle);

  // make a map detid-rechit
  std::map<DetId, const HGCRecHit*> hitmap;
  switch (algo_) {
    case 1: {
      iEvent.getByToken(recHitsEE_, recHitHandleEE);
      iEvent.getByToken(recHitsFH_, recHitHandleFH);
      iEvent.getByToken(recHitsBH_, recHitHandleBH);
      const auto& rechitsEE = *recHitHandleEE;
      const auto& rechitsFH = *recHitHandleFH;
      const auto& rechitsBH = *recHitHandleBH;
      for (unsigned int i = 0; i < rechitsEE.size(); ++i) {
        hitmap[rechitsEE[i].detid()] = &rechitsEE[i];
      }
      for (unsigned int i = 0; i < rechitsFH.size(); ++i) {
        hitmap[rechitsFH[i].detid()] = &rechitsFH[i];
      }
      for (unsigned int i = 0; i < rechitsBH.size(); ++i) {
        hitmap[rechitsBH[i].detid()] = &rechitsBH[i];
      }
      break;
    }
    case 2: {
      iEvent.getByToken(recHitsEE_, recHitHandleEE);
      const HGCRecHitCollection& rechitsEE = *recHitHandleEE;
      for (unsigned int i = 0; i < rechitsEE.size(); i++) {
        hitmap[rechitsEE[i].detid()] = &rechitsEE[i];
      }
      break;
    }
    case 3: {
      iEvent.getByToken(recHitsFH_, recHitHandleFH);
      iEvent.getByToken(recHitsBH_, recHitHandleBH);
      const auto& rechitsFH = *recHitHandleFH;
      const auto& rechitsBH = *recHitHandleBH;
      for (unsigned int i = 0; i < rechitsFH.size(); i++) {
        hitmap[rechitsFH[i].detid()] = &rechitsFH[i];
      }
      for (unsigned int i = 0; i < rechitsBH.size(); i++) {
        hitmap[rechitsBH[i].detid()] = &rechitsBH[i];
      }
      break;
    }
    default:
      assert(false);
      break;
  }

  // loop over caloParticles
  int seedDet = 0;
  float seedEnergy = 0.;
  IfLogTrace(debug_ > 0, "HGCalHitCalibration")
    << "Number of caloParticles: " << caloParticles.size() << std::endl;
  for (const auto& it_caloPart : caloParticles) {
    const SimClusterRefVector& simClusterRefVector = it_caloPart.simClusters();
    Energy_layer_calib_.fill(0.);
    Energy_layer_calib_fraction_.fill(0.);

    seedDet = 0;
    seedEnergy = 0.;
    for (const auto& it_sc : simClusterRefVector) {
      const SimCluster& simCluster = (*(it_sc));
      IfLogTrace(debug_ > 1, "HGCalHitCalibration")
	<< ">>> SC.energy(): " << simCluster.energy()
	<< " SC.simEnergy(): " << simCluster.simEnergy()
	<< std::endl;
      const std::vector<std::pair<uint32_t, float> >& hits_and_fractions =
          simCluster.hits_and_fractions();

      // loop over hits
      for (const auto& it_haf : hits_and_fractions) {
        unsigned int hitlayer = recHitTools_.getLayerWithOffset(it_haf.first);
        DetId hitid = (it_haf.first);
        // dump raw RecHits and match
        if (rawRecHits_) {
          if ((hitid.det() == DetId::Forward &&
              (hitid.subdetId() == HGCEE or hitid.subdetId() == HGCHEF or
               hitid.subdetId() == HGCHEB)) ||
	      (hitid.det() == DetId::Hcal && hitid.subdetId() == HcalEndcap))
            fillWithRecHits(hitmap, hitid, hitlayer, it_haf.second, seedDet,
                            seedEnergy);
        }
      }  // end simHit
    }    // end simCluster

    auto sumCalibRecHitCalib_fraction = std::accumulate(Energy_layer_calib_fraction_.begin(),
							Energy_layer_calib_fraction_.end(), 0.);
    IfLogTrace(debug_ > 0, "HGCalHitCalibration")
      << ">>> MC Energy: " << it_caloPart.energy()
      << " reco energy: " << sumCalibRecHitCalib_fraction
      << std::endl;
    if (h_EoP_CPene_calib_fraction_.count(seedDet))
      h_EoP_CPene_calib_fraction_[seedDet]->Fill(sumCalibRecHitCalib_fraction /
                                                 it_caloPart.energy());

    // Loop on reconstructed SC.
    const auto& clusters = *hgcalMultiClustersHandle;
    float total_energy = 0.;
    float max_dR2 = 0.0025;
    auto closest = std::min_element(clusters.begin(),
				    clusters.end(),
				    [&](const reco::PFCluster &a,
					const reco::PFCluster &b) {
	auto dR2_a = reco::deltaR2(it_caloPart, a);
	auto dR2_b = reco::deltaR2(it_caloPart, b);
	auto ERatio_a = a.correctedEnergy()/it_caloPart.energy();
	auto ERatio_b = b.correctedEnergy()/it_caloPart.energy();
	// If both clusters are within 0.0025, mark as first (min) the
	// element with the highest ratio against the SimCluster
	if (dR2_a < max_dR2 && dR2_b < max_dR2)
	  return ERatio_a > ERatio_b;
	return dR2_a < dR2_b;
      });
    if (closest != clusters.end()
	&& reco::deltaR2(*closest, it_caloPart) < 0.01) {
      total_energy = closest->correctedEnergy();
      seedDet = recHitTools_.getSiThickness(closest->seed());
      if (hgcal_EoP_CPene_calib_fraction_.count(seedDet)) {
	hgcal_EoP_CPene_calib_fraction_[seedDet]->Fill(total_energy /
						       it_caloPart.energy());
      }
    }

    auto closest_fcn = [&](auto const &a, auto const&b){
	auto dR2_a = reco::deltaR2(it_caloPart, a);
	auto dR2_b = reco::deltaR2(it_caloPart, b);
	auto ERatio_a = a.energy()/it_caloPart.energy();
	auto ERatio_b = b.energy()/it_caloPart.energy();
	// If both clusters are within 0.0025, mark as first (min) the
	// element with the highest ratio against the SimCluster
	if (dR2_a < max_dR2 && dR2_b < max_dR2)
	  return ERatio_a > ERatio_b;
	return dR2_a < dR2_b;
    };
    // ELECTRONS in HGCAL
    if (PFElectronHandle.isValid()) {
      auto const & ele = (*PFElectronHandle);
      auto closest = std::min_element(ele.begin(),
				      ele.end(),
				      closest_fcn);
      if (closest != ele.end()
	  && closest->superCluster()->seed()->seed().det() == DetId::Forward
	  && reco::deltaR2(*closest, it_caloPart) < 0.01) {
	seedDet = recHitTools_.getSiThickness(closest->superCluster()->seed()->seed());
	if (hgcal_ele_EoP_CPene_calib_fraction_.count(seedDet)) {
	  hgcal_ele_EoP_CPene_calib_fraction_[seedDet]->Fill(closest->energy() /
							     it_caloPart.energy());
	}
      }
    }

    // PHOTONS in HGCAL
    if (PFPhotonHandle.isValid()) {
      auto const & photon = (*PFPhotonHandle);
      auto closest = std::min_element(photon.begin(),
				      photon.end(),
				      closest_fcn);
      if (closest != photon.end()
	  && closest->superCluster()->seed()->seed().det() == DetId::Forward
	  && reco::deltaR2(*closest, it_caloPart) < 0.01) {
	seedDet = recHitTools_.getSiThickness(closest->superCluster()->seed()->seed());
	if (hgcal_photon_EoP_CPene_calib_fraction_.count(seedDet)) {
	  hgcal_photon_EoP_CPene_calib_fraction_[seedDet]->Fill(closest->energy() /
								it_caloPart.energy());
	}
      }
    }
  }  // end caloparticle
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void HGCalHitCalibration::fillDescriptions(
					   edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("debug", 0);
  desc.add<bool>("rawRecHits", true);
  desc.add<std::string>("detector", "all");
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("recHitsEE", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<edm::InputTag>("recHitsFH", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<edm::InputTag>("recHitsBH", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<edm::InputTag>("hgcalMultiClusters", edm::InputTag("particleFlowClusterHGCalFromMultiCl"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("ecalDrivenGsfElectronsFromMultiCl"));
  desc.add<edm::InputTag>("photons", edm::InputTag("photonsFromMultiCl"));
  descriptions.add("hgcalHitCalibration", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalHitCalibration);
