#include <numeric>
#include <iomanip>
#include <sstream>

#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "TMath.h"
#include "TLatex.h"
#include "TF1.h"

using namespace std;

//Parameters for the score cut. Later, this will become part of the
//configuration parameter for the HGCAL associator.
const double ScoreCutLCtoCP_ = 0.1;
const double ScoreCutCPtoLC_ = 0.1;
const double ScoreCutLCtoSC_ = 0.1;
const double ScoreCutSCtoLC_ = 0.1;
const double ScoreCutTStoSTSFakeMerge_[] = {0.6, FLT_MIN};  //1.e-09
const double ScoreCutSTStoTSPurDup_[] = {0.2, FLT_MIN};     //1.e-11

HGVHistoProducerAlgo::HGVHistoProducerAlgo(const edm::ParameterSet& pset)
    :  //parameters for eta
      minEta_(pset.getParameter<double>("minEta")),
      maxEta_(pset.getParameter<double>("maxEta")),
      nintEta_(pset.getParameter<int>("nintEta")),
      useFabsEta_(pset.getParameter<bool>("useFabsEta")),

      //parameters for energy
      minEne_(pset.getParameter<double>("minEne")),
      maxEne_(pset.getParameter<double>("maxEne")),
      nintEne_(pset.getParameter<int>("nintEne")),

      //parameters for pt
      minPt_(pset.getParameter<double>("minPt")),
      maxPt_(pset.getParameter<double>("maxPt")),
      nintPt_(pset.getParameter<int>("nintPt")),

      //parameters for phi
      minPhi_(pset.getParameter<double>("minPhi")),
      maxPhi_(pset.getParameter<double>("maxPhi")),
      nintPhi_(pset.getParameter<int>("nintPhi")),

      //parameters for counting mixed hits SimClusters
      minMixedHitsSimCluster_(pset.getParameter<double>("minMixedHitsSimCluster")),
      maxMixedHitsSimCluster_(pset.getParameter<double>("maxMixedHitsSimCluster")),
      nintMixedHitsSimCluster_(pset.getParameter<int>("nintMixedHitsSimCluster")),

      //parameters for counting mixed hits clusters
      minMixedHitsCluster_(pset.getParameter<double>("minMixedHitsCluster")),
      maxMixedHitsCluster_(pset.getParameter<double>("maxMixedHitsCluster")),
      nintMixedHitsCluster_(pset.getParameter<int>("nintMixedHitsCluster")),

      //parameters for the total amount of energy clustered by all layer clusters (fraction over CaloParticless)
      minEneCl_(pset.getParameter<double>("minEneCl")),
      maxEneCl_(pset.getParameter<double>("maxEneCl")),
      nintEneCl_(pset.getParameter<int>("nintEneCl")),

      //parameters for the longitudinal depth barycenter.
      minLongDepBary_(pset.getParameter<double>("minLongDepBary")),
      maxLongDepBary_(pset.getParameter<double>("maxLongDepBary")),
      nintLongDepBary_(pset.getParameter<int>("nintLongDepBary")),

      //parameters for z positionof vertex plots
      minZpos_(pset.getParameter<double>("minZpos")),
      maxZpos_(pset.getParameter<double>("maxZpos")),
      nintZpos_(pset.getParameter<int>("nintZpos")),

      //Parameters for the total number of SimClusters per layer
      minTotNsimClsperlay_(pset.getParameter<double>("minTotNsimClsperlay")),
      maxTotNsimClsperlay_(pset.getParameter<double>("maxTotNsimClsperlay")),
      nintTotNsimClsperlay_(pset.getParameter<int>("nintTotNsimClsperlay")),

      //Parameters for the total number of layer clusters per layer
      minTotNClsperlay_(pset.getParameter<double>("minTotNClsperlay")),
      maxTotNClsperlay_(pset.getParameter<double>("maxTotNClsperlay")),
      nintTotNClsperlay_(pset.getParameter<int>("nintTotNClsperlay")),

      //Parameters for the energy clustered by layer clusters per layer (fraction over CaloParticless)
      minEneClperlay_(pset.getParameter<double>("minEneClperlay")),
      maxEneClperlay_(pset.getParameter<double>("maxEneClperlay")),
      nintEneClperlay_(pset.getParameter<int>("nintEneClperlay")),

      //Parameters for the score both for:
      //1. calo particle to layer clusters association per layer
      //2. layer cluster to calo particles association per layer
      minScore_(pset.getParameter<double>("minScore")),
      maxScore_(pset.getParameter<double>("maxScore")),
      nintScore_(pset.getParameter<int>("nintScore")),

      //Parameters for shared energy fraction. That is:
      //1. Fraction of each of the layer clusters energy related to a
      //calo particle over that calo particle's energy.
      //2. Fraction of each of the calo particles energy
      //related to a layer cluster over that layer cluster's energy.
      minSharedEneFrac_(pset.getParameter<double>("minSharedEneFrac")),
      maxSharedEneFrac_(pset.getParameter<double>("maxSharedEneFrac")),
      nintSharedEneFrac_(pset.getParameter<int>("nintSharedEneFrac")),
      minTSTSharedEneFracEfficiency_(pset.getParameter<double>("minTSTSharedEneFracEfficiency")),

      //Same as above for Tracksters
      minTSTSharedEneFrac_(pset.getParameter<double>("minTSTSharedEneFrac")),
      maxTSTSharedEneFrac_(pset.getParameter<double>("maxTSTSharedEneFrac")),
      nintTSTSharedEneFrac_(pset.getParameter<int>("nintTSTSharedEneFrac")),

      //Parameters for the total number of SimClusters per thickness
      minTotNsimClsperthick_(pset.getParameter<double>("minTotNsimClsperthick")),
      maxTotNsimClsperthick_(pset.getParameter<double>("maxTotNsimClsperthick")),
      nintTotNsimClsperthick_(pset.getParameter<int>("nintTotNsimClsperthick")),

      //Parameters for the total number of layer clusters per thickness
      minTotNClsperthick_(pset.getParameter<double>("minTotNClsperthick")),
      maxTotNClsperthick_(pset.getParameter<double>("maxTotNClsperthick")),
      nintTotNClsperthick_(pset.getParameter<int>("nintTotNClsperthick")),

      //Parameters for the total number of cells per per thickness per layer
      minTotNcellsperthickperlayer_(pset.getParameter<double>("minTotNcellsperthickperlayer")),
      maxTotNcellsperthickperlayer_(pset.getParameter<double>("maxTotNcellsperthickperlayer")),
      nintTotNcellsperthickperlayer_(pset.getParameter<int>("nintTotNcellsperthickperlayer")),

      //Parameters for the distance of cluster cells to seed cell per thickness per layer
      minDisToSeedperthickperlayer_(pset.getParameter<double>("minDisToSeedperthickperlayer")),
      maxDisToSeedperthickperlayer_(pset.getParameter<double>("maxDisToSeedperthickperlayer")),
      nintDisToSeedperthickperlayer_(pset.getParameter<int>("nintDisToSeedperthickperlayer")),

      //Parameters for the energy weighted distance of cluster cells to seed cell per thickness per layer
      minDisToSeedperthickperlayerenewei_(pset.getParameter<double>("minDisToSeedperthickperlayerenewei")),
      maxDisToSeedperthickperlayerenewei_(pset.getParameter<double>("maxDisToSeedperthickperlayerenewei")),
      nintDisToSeedperthickperlayerenewei_(pset.getParameter<int>("nintDisToSeedperthickperlayerenewei")),

      //Parameters for the distance of cluster cells to max cell per thickness per layer
      minDisToMaxperthickperlayer_(pset.getParameter<double>("minDisToMaxperthickperlayer")),
      maxDisToMaxperthickperlayer_(pset.getParameter<double>("maxDisToMaxperthickperlayer")),
      nintDisToMaxperthickperlayer_(pset.getParameter<int>("nintDisToMaxperthickperlayer")),

      //Parameters for the energy weighted distance of cluster cells to max cell per thickness per layer
      minDisToMaxperthickperlayerenewei_(pset.getParameter<double>("minDisToMaxperthickperlayerenewei")),
      maxDisToMaxperthickperlayerenewei_(pset.getParameter<double>("maxDisToMaxperthickperlayerenewei")),
      nintDisToMaxperthickperlayerenewei_(pset.getParameter<int>("nintDisToMaxperthickperlayerenewei")),

      //Parameters for the distance of seed cell to max cell per thickness per layer
      minDisSeedToMaxperthickperlayer_(pset.getParameter<double>("minDisSeedToMaxperthickperlayer")),
      maxDisSeedToMaxperthickperlayer_(pset.getParameter<double>("maxDisSeedToMaxperthickperlayer")),
      nintDisSeedToMaxperthickperlayer_(pset.getParameter<int>("nintDisSeedToMaxperthickperlayer")),

      //Parameters for the energy of a cluster per thickness per layer
      minClEneperthickperlayer_(pset.getParameter<double>("minClEneperthickperlayer")),
      maxClEneperthickperlayer_(pset.getParameter<double>("maxClEneperthickperlayer")),
      nintClEneperthickperlayer_(pset.getParameter<int>("nintClEneperthickperlayer")),

      //Parameters for the energy density of cluster cells per thickness
      minCellsEneDensperthick_(pset.getParameter<double>("minCellsEneDensperthick")),
      maxCellsEneDensperthick_(pset.getParameter<double>("maxCellsEneDensperthick")),
      nintCellsEneDensperthick_(pset.getParameter<int>("nintCellsEneDensperthick")),

      //Parameters for the total number of Tracksters per event
      // Always treat one event as two events, one in +z one in -z
      minTotNTSTs_(pset.getParameter<double>("minTotNTSTs")),
      maxTotNTSTs_(pset.getParameter<double>("maxTotNTSTs")),
      nintTotNTSTs_(pset.getParameter<int>("nintTotNTSTs")),

      //Parameters for the total number of layer clusters in Trackster
      minTotNClsinTSTs_(pset.getParameter<double>("minTotNClsinTSTs")),
      maxTotNClsinTSTs_(pset.getParameter<double>("maxTotNClsinTSTs")),
      nintTotNClsinTSTs_(pset.getParameter<int>("nintTotNClsinTSTs")),

      //Parameters for the total number of layer clusters in Trackster per layer
      minTotNClsinTSTsperlayer_(pset.getParameter<double>("minTotNClsinTSTsperlayer")),
      maxTotNClsinTSTsperlayer_(pset.getParameter<double>("maxTotNClsinTSTsperlayer")),
      nintTotNClsinTSTsperlayer_(pset.getParameter<int>("nintTotNClsinTSTsperlayer")),

      //Parameters for the multiplicity of layer clusters in Trackster
      minMplofLCs_(pset.getParameter<double>("minMplofLCs")),
      maxMplofLCs_(pset.getParameter<double>("maxMplofLCs")),
      nintMplofLCs_(pset.getParameter<int>("nintMplofLCs")),

      //Parameters for cluster size
      minSizeCLsinTSTs_(pset.getParameter<double>("minSizeCLsinTSTs")),
      maxSizeCLsinTSTs_(pset.getParameter<double>("maxSizeCLsinTSTs")),
      nintSizeCLsinTSTs_(pset.getParameter<int>("nintSizeCLsinTSTs")),

      //Parameters for the energy of a cluster per thickness per layer
      minClEnepermultiplicity_(pset.getParameter<double>("minClEnepermultiplicity")),
      maxClEnepermultiplicity_(pset.getParameter<double>("maxClEnepermultiplicity")),
      nintClEnepermultiplicity_(pset.getParameter<int>("nintClEnepermultiplicity")),

      //parameters for x
      minX_(pset.getParameter<double>("minX")),
      maxX_(pset.getParameter<double>("maxX")),
      nintX_(pset.getParameter<int>("nintX")),

      //parameters for y
      minY_(pset.getParameter<double>("minY")),
      maxY_(pset.getParameter<double>("maxY")),
      nintY_(pset.getParameter<int>("nintY")),

      //parameters for z
      minZ_(pset.getParameter<double>("minZ")),
      maxZ_(pset.getParameter<double>("maxZ")),
      nintZ_(pset.getParameter<int>("nintZ")) {}

HGVHistoProducerAlgo::~HGVHistoProducerAlgo() {}

void HGVHistoProducerAlgo::bookInfo(DQMStore::IBooker& ibook, Histograms& histograms) {
  histograms.lastLayerEEzm = ibook.bookInt("lastLayerEEzm");
  histograms.lastLayerFHzm = ibook.bookInt("lastLayerFHzm");
  histograms.maxlayerzm = ibook.bookInt("maxlayerzm");
  histograms.lastLayerEEzp = ibook.bookInt("lastLayerEEzp");
  histograms.lastLayerFHzp = ibook.bookInt("lastLayerFHzp");
  histograms.maxlayerzp = ibook.bookInt("maxlayerzp");
}

void HGVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::IBooker& ibook,
                                                  Histograms& histograms,
                                                  int pdgid,
                                                  unsigned int layers) {
  histograms.h_caloparticle_eta[pdgid] =
      ibook.book1D("N of caloparticle vs eta", "N of caloParticles vs eta", nintEta_, minEta_, maxEta_);
  histograms.h_caloparticle_eta_Zorigin[pdgid] =
      ibook.book2D("Eta vs Zorigin", "Eta vs Zorigin", nintEta_, minEta_, maxEta_, nintZpos_, minZpos_, maxZpos_);

  histograms.h_caloparticle_energy[pdgid] =
      ibook.book1D("Energy", "Energy of CaloParticles; Energy [GeV]", nintEne_, minEne_, maxEne_);
  histograms.h_caloparticle_pt[pdgid] = ibook.book1D("Pt", "Pt of CaloParticles", nintPt_, minPt_, maxPt_);
  histograms.h_caloparticle_phi[pdgid] = ibook.book1D("Phi", "Phi of CaloParticles", nintPhi_, minPhi_, maxPhi_);
  histograms.h_caloparticle_selfenergy[pdgid] =
      ibook.book1D("SelfEnergy", "Total Energy of Hits in Sim Clusters (matched)", nintEne_, minEne_, maxEne_);
  histograms.h_caloparticle_energyDifference[pdgid] =
      ibook.book1D("EnergyDifference", "(Energy-SelfEnergy)/Energy", 300., -5., 1.);

  histograms.h_caloparticle_nSimClusters[pdgid] =
      ibook.book1D("Num Sim Clusters", "Num Sim Clusters in CaloParticles", 100, 0., 100.);
  histograms.h_caloparticle_nHitsInSimClusters[pdgid] =
      ibook.book1D("Num Hits in Sim Clusters", "Num Hits in Sim Clusters in CaloParticles", 1000, 0., 1000.);
  histograms.h_caloparticle_nHitsInSimClusters_matchedtoRecHit[pdgid] = ibook.book1D(
      "Num Rec-matched Hits in Sim Clusters", "Num Hits in Sim Clusters (matched) in CaloParticles", 1000, 0., 1000.);

  histograms.h_caloparticle_nHits_matched_energy[pdgid] =
      ibook.book1D("Energy of Rec-matched Hits", "Energy of Hits in Sim Clusters (matched)", 100, 0., 10.);
  histograms.h_caloparticle_nHits_matched_energy_layer[pdgid] =
      ibook.book2D("Energy of Rec-matched Hits vs layer",
                   "Energy of Hits in Sim Clusters (matched) vs layer",
                   2 * layers,
                   0.,
                   (float)2 * layers,
                   100,
                   0.,
                   10.);
  histograms.h_caloparticle_nHits_matched_energy_layer_1SimCl[pdgid] =
      ibook.book2D("Energy of Rec-matched Hits vs layer (1SC)",
                   "Energy of Hits only 1 Sim Clusters (matched) vs layer",
                   2 * layers,
                   0.,
                   (float)2 * layers,
                   100,
                   0.,
                   10.);
  histograms.h_caloparticle_sum_energy_layer[pdgid] =
      ibook.book2D("Rec-matched Hits Sum Energy vs layer",
                   "Rescaled Sum Energy of Hits in Sim Clusters (matched) vs layer",
                   2 * layers,
                   0.,
                   (float)2 * layers,
                   110,
                   0.,
                   110.);
  histograms.h_caloparticle_fractions[pdgid] =
      ibook.book2D("HitFractions", "Hit fractions;Hit fraction;E_{hit}^{2} fraction", 101, 0, 1.01, 100, 0, 1);
  histograms.h_caloparticle_fractions_weight[pdgid] = ibook.book2D(
      "HitFractions_weighted", "Hit fractions weighted;Hit fraction;E_{hit}^{2} fraction", 101, 0, 1.01, 100, 0, 1);

  histograms.h_caloparticle_firstlayer[pdgid] =
      ibook.book1D("First Layer", "First layer of the CaloParticles", 2 * layers, 0., (float)2 * layers);
  histograms.h_caloparticle_lastlayer[pdgid] =
      ibook.book1D("Last Layer", "Last layer of the CaloParticles", 2 * layers, 0., (float)2 * layers);
  histograms.h_caloparticle_layersnum[pdgid] =
      ibook.book1D("Number of Layers", "Number of layers of the CaloParticles", 2 * layers, 0., (float)2 * layers);
  histograms.h_caloparticle_firstlayer_matchedtoRecHit[pdgid] = ibook.book1D(
      "First Layer (rec-matched hit)", "First layer of the CaloParticles (matched)", 2 * layers, 0., (float)2 * layers);
  histograms.h_caloparticle_lastlayer_matchedtoRecHit[pdgid] = ibook.book1D(
      "Last Layer (rec-matched hit)", "Last layer of the CaloParticles (matched)", 2 * layers, 0., (float)2 * layers);
  histograms.h_caloparticle_layersnum_matchedtoRecHit[pdgid] =
      ibook.book1D("Number of Layers (rec-matched hit)",
                   "Number of layers of the CaloParticles (matched)",
                   2 * layers,
                   0.,
                   (float)2 * layers);
}

void HGVHistoProducerAlgo::bookSimClusterHistos(DQMStore::IBooker& ibook,
                                                Histograms& histograms,
                                                unsigned int layers,
                                                std::vector<int> thicknesses) {
  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    // Make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    // first with the -z endcap
    if (ilayer < layers) {
      istr2 = std::to_string(ilayer + 1) + " in z-";
    } else {  // then for the +z
      istr2 = std::to_string(ilayer - (layers - 1)) + " in z+";
    }
    histograms.h_simclusternum_perlayer[ilayer] = ibook.book1D("totsimclusternum_layer_" + istr1,
                                                               "total number of SimClusters for layer " + istr2,
                                                               nintTotNsimClsperlay_,
                                                               minTotNsimClsperlay_,
                                                               maxTotNsimClsperlay_);

  }  //end of loop over layers
  //---------------------------------------------------------------------------------------------------------------------------
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    auto istr = std::to_string(*it);
    histograms.h_simclusternum_perthick[(*it)] = ibook.book1D("totsimclusternum_thick_" + istr,
                                                              "total number of simclusters for thickness " + istr,
                                                              nintTotNsimClsperthick_,
                                                              minTotNsimClsperthick_,
                                                              maxTotNsimClsperthick_);
  }  //end of loop over thicknesses

  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_mixedhitssimcluster_zminus =
      ibook.book1D("mixedhitssimcluster_zminus",
                   "N of simclusters that contain hits of more than one kind in z-",
                   nintMixedHitsSimCluster_,
                   minMixedHitsSimCluster_,
                   maxMixedHitsSimCluster_);
  //z+
  histograms.h_mixedhitssimcluster_zplus =
      ibook.book1D("mixedhitssimcluster_zplus",
                   "N of simclusters that contain hits of more than one kind in z+",
                   nintMixedHitsSimCluster_,
                   minMixedHitsSimCluster_,
                   maxMixedHitsSimCluster_);
}

void HGVHistoProducerAlgo::bookSimClusterAssociationHistos(DQMStore::IBooker& ibook,
                                                           Histograms& histograms,
                                                           unsigned int layers,
                                                           std::vector<int> thicknesses) {
  std::unordered_map<int, dqm::reco::MonitorElement*> denom_layercl_in_simcl_eta_perlayer;
  denom_layercl_in_simcl_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> denom_layercl_in_simcl_phi_perlayer;
  denom_layercl_in_simcl_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> score_layercl2simcluster_perlayer;
  score_layercl2simcluster_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> sharedenergy_layercl2simcluster_perlayer;
  sharedenergy_layercl2simcluster_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> energy_vs_score_layercl2simcluster_perlayer;
  energy_vs_score_layercl2simcluster_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> num_layercl_in_simcl_eta_perlayer;
  num_layercl_in_simcl_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> num_layercl_in_simcl_phi_perlayer;
  num_layercl_in_simcl_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> numMerge_layercl_in_simcl_eta_perlayer;
  numMerge_layercl_in_simcl_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> numMerge_layercl_in_simcl_phi_perlayer;
  numMerge_layercl_in_simcl_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> sharedenergy_layercl2simcluster_vs_eta_perlayer;
  sharedenergy_layercl2simcluster_vs_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> sharedenergy_layercl2simcluster_vs_phi_perlayer;
  sharedenergy_layercl2simcluster_vs_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> denom_simcluster_eta_perlayer;
  denom_simcluster_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> denom_simcluster_phi_perlayer;
  denom_simcluster_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> score_simcluster2layercl_perlayer;
  score_simcluster2layercl_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> sharedenergy_simcluster2layercl_perlayer;
  sharedenergy_simcluster2layercl_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> energy_vs_score_simcluster2layercl_perlayer;
  energy_vs_score_simcluster2layercl_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> num_simcluster_eta_perlayer;
  num_simcluster_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> num_simcluster_phi_perlayer;
  num_simcluster_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> numDup_simcluster_eta_perlayer;
  numDup_simcluster_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> numDup_simcluster_phi_perlayer;
  numDup_simcluster_phi_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> sharedenergy_simcluster2layercl_vs_eta_perlayer;
  sharedenergy_simcluster2layercl_vs_eta_perlayer.clear();
  std::unordered_map<int, dqm::reco::MonitorElement*> sharedenergy_simcluster2layercl_vs_phi_perlayer;
  sharedenergy_simcluster2layercl_vs_phi_perlayer.clear();

  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    // Make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    // first with the -z endcap
    if (ilayer < layers) {
      istr2 = std::to_string(ilayer + 1) + " in z-";
    } else {  // then for the +z
      istr2 = std::to_string(ilayer - (layers - 1)) + " in z+";
    }
    //-------------------------------------------------------------------------------------------------------------------------
    denom_layercl_in_simcl_eta_perlayer[ilayer] =
        ibook.book1D("Denom_LayerCluster_in_SimCluster_Eta_perlayer" + istr1,
                     "Denom LayerCluster in SimCluster Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    //-------------------------------------------------------------------------------------------------------------------------
    denom_layercl_in_simcl_phi_perlayer[ilayer] =
        ibook.book1D("Denom_LayerCluster_in_SimCluster_Phi_perlayer" + istr1,
                     "Denom LayerCluster in SimCluster Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    //-------------------------------------------------------------------------------------------------------------------------
    score_layercl2simcluster_perlayer[ilayer] = ibook.book1D("Score_layercl2simcluster_perlayer" + istr1,
                                                             "Score of Layer Cluster per SimCluster for layer " + istr2,
                                                             nintScore_,
                                                             minScore_,
                                                             maxScore_);
    //-------------------------------------------------------------------------------------------------------------------------
    score_simcluster2layercl_perlayer[ilayer] = ibook.book1D("Score_simcluster2layercl_perlayer" + istr1,
                                                             "Score of SimCluster per Layer Cluster for layer " + istr2,
                                                             nintScore_,
                                                             minScore_,
                                                             maxScore_);
    //-------------------------------------------------------------------------------------------------------------------------
    energy_vs_score_simcluster2layercl_perlayer[ilayer] =
        ibook.book2D("Energy_vs_Score_simcluster2layer_perlayer" + istr1,
                     "Energy vs Score of SimCluster per Layer Cluster for layer " + istr2,
                     nintScore_,
                     minScore_,
                     maxScore_,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    energy_vs_score_layercl2simcluster_perlayer[ilayer] =
        ibook.book2D("Energy_vs_Score_layer2simcluster_perlayer" + istr1,
                     "Energy vs Score of Layer Cluster per SimCluster for layer " + istr2,
                     nintScore_,
                     minScore_,
                     maxScore_,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    sharedenergy_simcluster2layercl_perlayer[ilayer] =
        ibook.book1D("SharedEnergy_simcluster2layercl_perlayer" + istr1,
                     "Shared Energy of SimCluster per Layer Cluster for layer " + istr2,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    sharedenergy_simcluster2layercl_vs_eta_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_simcluster2layercl_vs_eta_perlayer" + istr1,
                          "Shared Energy of SimCluster vs #eta per best Layer Cluster for layer " + istr2,
                          nintEta_,
                          minEta_,
                          maxEta_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    sharedenergy_simcluster2layercl_vs_phi_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_simcluster2layercl_vs_phi_perlayer" + istr1,
                          "Shared Energy of SimCluster vs #phi per best Layer Cluster for layer " + istr2,
                          nintPhi_,
                          minPhi_,
                          maxPhi_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    sharedenergy_layercl2simcluster_perlayer[ilayer] =
        ibook.book1D("SharedEnergy_layercluster2simcluster_perlayer" + istr1,
                     "Shared Energy of Layer Cluster per SimCluster for layer " + istr2,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    sharedenergy_layercl2simcluster_vs_eta_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_layercl2simcluster_vs_eta_perlayer" + istr1,
                          "Shared Energy of LayerCluster vs #eta per best SimCluster for layer " + istr2,
                          nintEta_,
                          minEta_,
                          maxEta_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    sharedenergy_layercl2simcluster_vs_phi_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_layercl2simcluster_vs_phi_perlayer" + istr1,
                          "Shared Energy of LayerCluster vs #phi per best SimCluster for layer " + istr2,
                          nintPhi_,
                          minPhi_,
                          maxPhi_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    //-------------------------------------------------------------------------------------------------------------------------
    num_simcluster_eta_perlayer[ilayer] = ibook.book1D("Num_SimCluster_Eta_perlayer" + istr1,
                                                       "Num SimCluster Eta per Layer Cluster for layer " + istr2,
                                                       nintEta_,
                                                       minEta_,
                                                       maxEta_);
    //-------------------------------------------------------------------------------------------------------------------------
    numDup_simcluster_eta_perlayer[ilayer] =
        ibook.book1D("NumDup_SimCluster_Eta_perlayer" + istr1,
                     "Num Duplicate SimCluster Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    //-------------------------------------------------------------------------------------------------------------------------
    denom_simcluster_eta_perlayer[ilayer] = ibook.book1D("Denom_SimCluster_Eta_perlayer" + istr1,
                                                         "Denom SimCluster Eta per Layer Cluster for layer " + istr2,
                                                         nintEta_,
                                                         minEta_,
                                                         maxEta_);
    //-------------------------------------------------------------------------------------------------------------------------
    num_simcluster_phi_perlayer[ilayer] = ibook.book1D("Num_SimCluster_Phi_perlayer" + istr1,
                                                       "Num SimCluster Phi per Layer Cluster for layer " + istr2,
                                                       nintPhi_,
                                                       minPhi_,
                                                       maxPhi_);
    //-------------------------------------------------------------------------------------------------------------------------
    numDup_simcluster_phi_perlayer[ilayer] =
        ibook.book1D("NumDup_SimCluster_Phi_perlayer" + istr1,
                     "Num Duplicate SimCluster Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    //-------------------------------------------------------------------------------------------------------------------------
    denom_simcluster_phi_perlayer[ilayer] = ibook.book1D("Denom_SimCluster_Phi_perlayer" + istr1,
                                                         "Denom SimCluster Phi per Layer Cluster for layer " + istr2,
                                                         nintPhi_,
                                                         minPhi_,
                                                         maxPhi_);
    //-------------------------------------------------------------------------------------------------------------------------
    num_layercl_in_simcl_eta_perlayer[ilayer] =
        ibook.book1D("Num_LayerCluster_in_SimCluster_Eta_perlayer" + istr1,
                     "Num LayerCluster Eta per Layer Cluster in SimCluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    //-------------------------------------------------------------------------------------------------------------------------
    numMerge_layercl_in_simcl_eta_perlayer[ilayer] =
        ibook.book1D("NumMerge_LayerCluster_in_SimCluster_Eta_perlayer" + istr1,
                     "Num Merge LayerCluster Eta per Layer Cluster in SimCluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    //-------------------------------------------------------------------------------------------------------------------------
    num_layercl_in_simcl_phi_perlayer[ilayer] =
        ibook.book1D("Num_LayerCluster_in_SimCluster_Phi_perlayer" + istr1,
                     "Num LayerCluster Phi per Layer Cluster in SimCluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    //-------------------------------------------------------------------------------------------------------------------------
    numMerge_layercl_in_simcl_phi_perlayer[ilayer] =
        ibook.book1D("NumMerge_LayerCluster_in_SimCluster_Phi_perlayer" + istr1,
                     "Num Merge LayerCluster Phi per Layer Cluster in SimCluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);

  }  //end of loop over layers

  histograms.h_denom_layercl_in_simcl_eta_perlayer.push_back(std::move(denom_layercl_in_simcl_eta_perlayer));
  histograms.h_denom_layercl_in_simcl_phi_perlayer.push_back(std::move(denom_layercl_in_simcl_phi_perlayer));
  histograms.h_score_layercl2simcluster_perlayer.push_back(std::move(score_layercl2simcluster_perlayer));
  histograms.h_sharedenergy_layercl2simcluster_perlayer.push_back(std::move(sharedenergy_layercl2simcluster_perlayer));
  histograms.h_energy_vs_score_layercl2simcluster_perlayer.push_back(
      std::move(energy_vs_score_layercl2simcluster_perlayer));
  histograms.h_num_layercl_in_simcl_eta_perlayer.push_back(std::move(num_layercl_in_simcl_eta_perlayer));
  histograms.h_num_layercl_in_simcl_phi_perlayer.push_back(std::move(num_layercl_in_simcl_phi_perlayer));
  histograms.h_numMerge_layercl_in_simcl_eta_perlayer.push_back(std::move(numMerge_layercl_in_simcl_eta_perlayer));
  histograms.h_numMerge_layercl_in_simcl_phi_perlayer.push_back(std::move(numMerge_layercl_in_simcl_phi_perlayer));
  histograms.h_sharedenergy_layercl2simcluster_vs_eta_perlayer.push_back(
      std::move(sharedenergy_layercl2simcluster_vs_eta_perlayer));
  histograms.h_sharedenergy_layercl2simcluster_vs_phi_perlayer.push_back(
      std::move(sharedenergy_layercl2simcluster_vs_phi_perlayer));
  histograms.h_denom_simcluster_eta_perlayer.push_back(std::move(denom_simcluster_eta_perlayer));
  histograms.h_denom_simcluster_phi_perlayer.push_back(std::move(denom_simcluster_phi_perlayer));
  histograms.h_score_simcluster2layercl_perlayer.push_back(std::move(score_simcluster2layercl_perlayer));
  histograms.h_sharedenergy_simcluster2layercl_perlayer.push_back(std::move(sharedenergy_simcluster2layercl_perlayer));
  histograms.h_energy_vs_score_simcluster2layercl_perlayer.push_back(
      std::move(energy_vs_score_simcluster2layercl_perlayer));
  histograms.h_num_simcluster_eta_perlayer.push_back(std::move(num_simcluster_eta_perlayer));
  histograms.h_num_simcluster_phi_perlayer.push_back(std::move(num_simcluster_phi_perlayer));
  histograms.h_numDup_simcluster_eta_perlayer.push_back(std::move(numDup_simcluster_eta_perlayer));
  histograms.h_numDup_simcluster_phi_perlayer.push_back(std::move(numDup_simcluster_phi_perlayer));
  histograms.h_sharedenergy_simcluster2layercl_vs_eta_perlayer.push_back(
      std::move(sharedenergy_simcluster2layercl_vs_eta_perlayer));
  histograms.h_sharedenergy_simcluster2layercl_vs_phi_perlayer.push_back(
      std::move(sharedenergy_simcluster2layercl_vs_phi_perlayer));
}
void HGVHistoProducerAlgo::bookClusterHistos_ClusterLevel(DQMStore::IBooker& ibook,
                                                          Histograms& histograms,
                                                          unsigned int layers,
                                                          std::vector<int> thicknesses,
                                                          std::string pathtomatbudfile) {
  //---------------------------------------------------------------------------------------------------------------------------
  histograms.h_cluster_eta.push_back(
      ibook.book1D("num_reco_cluster_eta", "N of reco clusters vs eta", nintEta_, minEta_, maxEta_));

  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_mixedhitscluster_zminus.push_back(
      ibook.book1D("mixedhitscluster_zminus",
                   "N of reco clusters that contain hits of more than one kind in z-",
                   nintMixedHitsCluster_,
                   minMixedHitsCluster_,
                   maxMixedHitsCluster_));
  //z+
  histograms.h_mixedhitscluster_zplus.push_back(
      ibook.book1D("mixedhitscluster_zplus",
                   "N of reco clusters that contain hits of more than one kind in z+",
                   nintMixedHitsCluster_,
                   minMixedHitsCluster_,
                   maxMixedHitsCluster_));

  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_energyclustered_zminus.push_back(
      ibook.book1D("energyclustered_zminus",
                   "percent of total energy clustered by all layer clusters over CaloParticless energy in z-",
                   nintEneCl_,
                   minEneCl_,
                   maxEneCl_));
  //z+
  histograms.h_energyclustered_zplus.push_back(
      ibook.book1D("energyclustered_zplus",
                   "percent of total energy clustered by all layer clusters over CaloParticless energy in z+",
                   nintEneCl_,
                   minEneCl_,
                   maxEneCl_));

  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  std::string subpathtomat = pathtomatbudfile.substr(pathtomatbudfile.find("Validation"));
  histograms.h_longdepthbarycentre_zminus.push_back(
      ibook.book1D("longdepthbarycentre_zminus",
                   "The longitudinal depth barycentre in z- for " + subpathtomat,
                   nintLongDepBary_,
                   minLongDepBary_,
                   maxLongDepBary_));
  //z+
  histograms.h_longdepthbarycentre_zplus.push_back(
      ibook.book1D("longdepthbarycentre_zplus",
                   "The longitudinal depth barycentre in z+ for " + subpathtomat,
                   nintLongDepBary_,
                   minLongDepBary_,
                   maxLongDepBary_));

  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    // Make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    // first with the -z endcap
    if (ilayer < layers) {
      istr2 = std::to_string(ilayer + 1) + " in z-";
    } else {  // then for the +z
      istr2 = std::to_string(ilayer - (layers - 1)) + " in z+";
    }
    histograms.h_clusternum_perlayer[ilayer] = ibook.book1D("totclusternum_layer_" + istr1,
                                                            "total number of layer clusters for layer " + istr2,
                                                            nintTotNClsperlay_,
                                                            minTotNClsperlay_,
                                                            maxTotNClsperlay_);
    histograms.h_energyclustered_perlayer[ilayer] = ibook.book1D(
        "energyclustered_perlayer" + istr1,
        "percent of total energy clustered by layer clusters over CaloParticless energy for layer " + istr2,
        nintEneClperlay_,
        minEneClperlay_,
        maxEneClperlay_);
  }

  //---------------------------------------------------------------------------------------------------------------------------
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    auto istr = std::to_string(*it);
    histograms.h_clusternum_perthick[(*it)] = ibook.book1D("totclusternum_thick_" + istr,
                                                           "total number of layer clusters for thickness " + istr,
                                                           nintTotNClsperthick_,
                                                           minTotNClsperthick_,
                                                           maxTotNClsperthick_);
  }
  //---------------------------------------------------------------------------------------------------------------------------
}

void HGVHistoProducerAlgo::bookClusterHistos_LCtoCP_association(DQMStore::IBooker& ibook,
                                                                Histograms& histograms,
                                                                unsigned int layers) {
  //----------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    // Make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    // first with the -z endcap
    if (ilayer < layers) {
      istr2 = std::to_string(ilayer + 1) + " in z-";
    } else {  // then for the +z
      istr2 = std::to_string(ilayer - (layers - 1)) + " in z+";
    }
    histograms.h_score_layercl2caloparticle_perlayer[ilayer] =
        ibook.book1D("Score_layercl2caloparticle_perlayer" + istr1,
                     "Score of Layer Cluster per CaloParticle for layer " + istr2,
                     nintScore_,
                     minScore_,
                     maxScore_);
    histograms.h_score_caloparticle2layercl_perlayer[ilayer] =
        ibook.book1D("Score_caloparticle2layercl_perlayer" + istr1,
                     "Score of CaloParticle per Layer Cluster for layer " + istr2,
                     nintScore_,
                     minScore_,
                     maxScore_);
    histograms.h_energy_vs_score_caloparticle2layercl_perlayer[ilayer] =
        ibook.book2D("Energy_vs_Score_caloparticle2layer_perlayer" + istr1,
                     "Energy vs Score of CaloParticle per Layer Cluster for layer " + istr2,
                     nintScore_,
                     minScore_,
                     maxScore_,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    histograms.h_energy_vs_score_layercl2caloparticle_perlayer[ilayer] =
        ibook.book2D("Energy_vs_Score_layer2caloparticle_perlayer" + istr1,
                     "Energy vs Score of Layer Cluster per CaloParticle Layer for layer " + istr2,
                     nintScore_,
                     minScore_,
                     maxScore_,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    histograms.h_sharedenergy_caloparticle2layercl_perlayer[ilayer] =
        ibook.book1D("SharedEnergy_caloparticle2layercl_perlayer" + istr1,
                     "Shared Energy of CaloParticle per Layer Cluster for layer " + istr2,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    histograms.h_sharedenergy_caloparticle2layercl_vs_eta_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_caloparticle2layercl_vs_eta_perlayer" + istr1,
                          "Shared Energy of CaloParticle vs #eta per best Layer Cluster for layer " + istr2,
                          nintEta_,
                          minEta_,
                          maxEta_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    histograms.h_sharedenergy_caloparticle2layercl_vs_phi_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_caloparticle2layercl_vs_phi_perlayer" + istr1,
                          "Shared Energy of CaloParticle vs #phi per best Layer Cluster for layer " + istr2,
                          nintPhi_,
                          minPhi_,
                          maxPhi_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    histograms.h_sharedenergy_layercl2caloparticle_perlayer[ilayer] =
        ibook.book1D("SharedEnergy_layercluster2caloparticle_perlayer" + istr1,
                     "Shared Energy of Layer Cluster per Layer Calo Particle for layer " + istr2,
                     nintSharedEneFrac_,
                     minSharedEneFrac_,
                     maxSharedEneFrac_);
    histograms.h_sharedenergy_layercl2caloparticle_vs_eta_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_layercl2caloparticle_vs_eta_perlayer" + istr1,
                          "Shared Energy of LayerCluster vs #eta per best Calo Particle for layer " + istr2,
                          nintEta_,
                          minEta_,
                          maxEta_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    histograms.h_sharedenergy_layercl2caloparticle_vs_phi_perlayer[ilayer] =
        ibook.bookProfile("SharedEnergy_layercl2caloparticle_vs_phi_perlayer" + istr1,
                          "Shared Energy of LayerCluster vs #phi per best Calo Particle for layer " + istr2,
                          nintPhi_,
                          minPhi_,
                          maxPhi_,
                          minSharedEneFrac_,
                          maxSharedEneFrac_);
    histograms.h_num_caloparticle_eta_perlayer[ilayer] =
        ibook.book1D("Num_CaloParticle_Eta_perlayer" + istr1,
                     "Num CaloParticle Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    histograms.h_numDup_caloparticle_eta_perlayer[ilayer] =
        ibook.book1D("NumDup_CaloParticle_Eta_perlayer" + istr1,
                     "Num Duplicate CaloParticle Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    histograms.h_denom_caloparticle_eta_perlayer[ilayer] =
        ibook.book1D("Denom_CaloParticle_Eta_perlayer" + istr1,
                     "Denom CaloParticle Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    histograms.h_num_caloparticle_phi_perlayer[ilayer] =
        ibook.book1D("Num_CaloParticle_Phi_perlayer" + istr1,
                     "Num CaloParticle Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    histograms.h_numDup_caloparticle_phi_perlayer[ilayer] =
        ibook.book1D("NumDup_CaloParticle_Phi_perlayer" + istr1,
                     "Num Duplicate CaloParticle Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    histograms.h_denom_caloparticle_phi_perlayer[ilayer] =
        ibook.book1D("Denom_CaloParticle_Phi_perlayer" + istr1,
                     "Denom CaloParticle Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    histograms.h_num_layercl_eta_perlayer[ilayer] =
        ibook.book1D("Num_LayerCluster_Eta_perlayer" + istr1,
                     "Num LayerCluster Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    histograms.h_numMerge_layercl_eta_perlayer[ilayer] =
        ibook.book1D("NumMerge_LayerCluster_Eta_perlayer" + istr1,
                     "Num Merge LayerCluster Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    histograms.h_denom_layercl_eta_perlayer[ilayer] =
        ibook.book1D("Denom_LayerCluster_Eta_perlayer" + istr1,
                     "Denom LayerCluster Eta per Layer Cluster for layer " + istr2,
                     nintEta_,
                     minEta_,
                     maxEta_);
    histograms.h_num_layercl_phi_perlayer[ilayer] =
        ibook.book1D("Num_LayerCluster_Phi_perlayer" + istr1,
                     "Num LayerCluster Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    histograms.h_numMerge_layercl_phi_perlayer[ilayer] =
        ibook.book1D("NumMerge_LayerCluster_Phi_perlayer" + istr1,
                     "Num Merge LayerCluster Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
    histograms.h_denom_layercl_phi_perlayer[ilayer] =
        ibook.book1D("Denom_LayerCluster_Phi_perlayer" + istr1,
                     "Denom LayerCluster Phi per Layer Cluster for layer " + istr2,
                     nintPhi_,
                     minPhi_,
                     maxPhi_);
  }
  //---------------------------------------------------------------------------------------------------------------------------
}

void HGVHistoProducerAlgo::bookClusterHistos_CellLevel(DQMStore::IBooker& ibook,
                                                       Histograms& histograms,
                                                       unsigned int layers,
                                                       std::vector<int> thicknesses) {
  //----------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    // Make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    // first with the -z endcap
    if (ilayer < layers) {
      istr2 = std::to_string(ilayer + 1) + " in z-";
    } else {  // then for the +z
      istr2 = std::to_string(ilayer - (layers - 1)) + " in z+";
    }
    histograms.h_cellAssociation_perlayer[ilayer] =
        ibook.book1D("cellAssociation_perlayer" + istr1, "Cell Association for layer " + istr2, 5, -4., 1.);
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(2, "TN(purity)");
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(3, "FN(ineff.)");
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(4, "FP(fake)");
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(5, "TP(eff.)");
  }
  //----------------------------------------------------------------------------------------------------------------------------
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    auto istr = std::to_string(*it);
    histograms.h_cellsenedens_perthick[(*it)] = ibook.book1D("cellsenedens_thick_" + istr,
                                                             "energy density of cluster cells for thickness " + istr,
                                                             nintCellsEneDensperthick_,
                                                             minCellsEneDensperthick_,
                                                             maxCellsEneDensperthick_);
  }
  //----------------------------------------------------------------------------------------------------------------------------
  //Not all combination exists but should keep them all for cross checking reason.
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
      auto istr1 = std::to_string(*it);
      auto istr2 = std::to_string(ilayer);
      while (istr2.size() < 2)
        istr2.insert(0, "0");
      auto istr = istr1 + "_" + istr2;
      // Make a mapping to the regural layer naming plus z- or z+ for convenience
      std::string istr3 = "";
      // first with the -z endcap
      if (ilayer < layers) {
        istr3 = std::to_string(ilayer + 1) + " in z- ";
      } else {  // then for the +z
        istr3 = std::to_string(ilayer - (layers - 1)) + " in z+ ";
      }
      //---
      histograms.h_cellsnum_perthickperlayer[istr] =
          ibook.book1D("cellsnum_perthick_perlayer_" + istr,
                       "total number of cells for layer " + istr3 + " for thickness " + istr1,
                       nintTotNcellsperthickperlayer_,
                       minTotNcellsperthickperlayer_,
                       maxTotNcellsperthickperlayer_);
      //---
      histograms.h_distancetoseedcell_perthickperlayer[istr] =
          ibook.book1D("distancetoseedcell_perthickperlayer_" + istr,
                       "distance of cluster cells to seed cell for layer " + istr3 + " for thickness " + istr1,
                       nintDisToSeedperthickperlayer_,
                       minDisToSeedperthickperlayer_,
                       maxDisToSeedperthickperlayer_);
      //---
      histograms.h_distancetoseedcell_perthickperlayer_eneweighted[istr] = ibook.book1D(
          "distancetoseedcell_perthickperlayer_eneweighted_" + istr,
          "energy weighted distance of cluster cells to seed cell for layer " + istr3 + " for thickness " + istr1,
          nintDisToSeedperthickperlayerenewei_,
          minDisToSeedperthickperlayerenewei_,
          maxDisToSeedperthickperlayerenewei_);
      //---
      histograms.h_distancetomaxcell_perthickperlayer[istr] =
          ibook.book1D("distancetomaxcell_perthickperlayer_" + istr,
                       "distance of cluster cells to max cell for layer " + istr3 + " for thickness " + istr1,
                       nintDisToMaxperthickperlayer_,
                       minDisToMaxperthickperlayer_,
                       maxDisToMaxperthickperlayer_);
      //---
      histograms.h_distancetomaxcell_perthickperlayer_eneweighted[istr] = ibook.book1D(
          "distancetomaxcell_perthickperlayer_eneweighted_" + istr,
          "energy weighted distance of cluster cells to max cell for layer " + istr3 + " for thickness " + istr1,
          nintDisToMaxperthickperlayerenewei_,
          minDisToMaxperthickperlayerenewei_,
          maxDisToMaxperthickperlayerenewei_);
      //---
      histograms.h_distancebetseedandmaxcell_perthickperlayer[istr] =
          ibook.book1D("distancebetseedandmaxcell_perthickperlayer_" + istr,
                       "distance of seed cell to max cell for layer " + istr3 + " for thickness " + istr1,
                       nintDisSeedToMaxperthickperlayer_,
                       minDisSeedToMaxperthickperlayer_,
                       maxDisSeedToMaxperthickperlayer_);
      //---
      histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer[istr] = ibook.book2D(
          "distancebetseedandmaxcellvsclusterenergy_perthickperlayer_" + istr,
          "distance of seed cell to max cell vs cluster energy for layer " + istr3 + " for thickness " + istr1,
          nintDisSeedToMaxperthickperlayer_,
          minDisSeedToMaxperthickperlayer_,
          maxDisSeedToMaxperthickperlayer_,
          nintClEneperthickperlayer_,
          minClEneperthickperlayer_,
          maxClEneperthickperlayer_);
    }
  }
}
//----------------------------------------------------------------------------------------------------------------------------

void HGVHistoProducerAlgo::bookTracksterHistos(DQMStore::IBooker& ibook, Histograms& histograms, unsigned int layers) {
  std::unordered_map<int, dqm::reco::MonitorElement*> clusternum_in_trackster_perlayer;
  clusternum_in_trackster_perlayer.clear();

  for (unsigned ilayer = 0; ilayer < 2 * layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while (istr1.size() < 2) {
      istr1.insert(0, "0");
    }
    // Make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    // first with the -z endcap
    if (ilayer < layers) {
      istr2 = std::to_string(ilayer + 1) + " in z-";
    } else {  // then for the +z
      istr2 = std::to_string(ilayer - (layers - 1)) + " in z+";
    }

    clusternum_in_trackster_perlayer[ilayer] = ibook.book1D("clusternum_in_trackster_perlayer" + istr1,
                                                            "Number of layer clusters in Trackster for layer " + istr2,
                                                            nintTotNClsinTSTsperlayer_,
                                                            minTotNClsinTSTsperlayer_,
                                                            maxTotNClsinTSTsperlayer_);
  }

  histograms.h_clusternum_in_trackster_perlayer.push_back(std::move(clusternum_in_trackster_perlayer));

  histograms.h_tracksternum.push_back(ibook.book1D(
      "tottracksternum", "total number of Tracksters;# of Tracksters", nintTotNTSTs_, minTotNTSTs_, maxTotNTSTs_));

  histograms.h_conttracksternum.push_back(ibook.book1D(
      "conttracksternum", "number of Tracksters with 3 contiguous layers", nintTotNTSTs_, minTotNTSTs_, maxTotNTSTs_));

  histograms.h_nonconttracksternum.push_back(ibook.book1D("nonconttracksternum",
                                                          "number of Tracksters without 3 contiguous layers",
                                                          nintTotNTSTs_,
                                                          minTotNTSTs_,
                                                          maxTotNTSTs_));

  histograms.h_clusternum_in_trackster.push_back(
      ibook.book1D("clusternum_in_trackster",
                   "total number of layer clusters in Trackster;# of LayerClusters",
                   nintTotNClsinTSTs_,
                   minTotNClsinTSTs_,
                   maxTotNClsinTSTs_));

  histograms.h_clusternum_in_trackster_vs_layer.push_back(ibook.bookProfile(
      "clusternum_in_trackster_vs_layer",
      "Profile of 2d layer clusters in Trackster vs layer number;layer number;<2D LayerClusters in Trackster>",
      2 * layers,
      0.,
      2. * layers,
      minTotNClsinTSTsperlayer_,
      maxTotNClsinTSTsperlayer_));

  histograms.h_multiplicityOfLCinTST.push_back(
      ibook.book2D("multiplicityOfLCinTST",
                   "Multiplicity vs Layer cluster size in Tracksters;LayerCluster multiplicity in Tracksters;Cluster "
                   "size (n_{hits})",
                   nintMplofLCs_,
                   minMplofLCs_,
                   maxMplofLCs_,
                   nintSizeCLsinTSTs_,
                   minSizeCLsinTSTs_,
                   maxSizeCLsinTSTs_));

  histograms.h_multiplicity_numberOfEventsHistogram.push_back(ibook.book1D("multiplicity_numberOfEventsHistogram",
                                                                           "multiplicity numberOfEventsHistogram",
                                                                           nintMplofLCs_,
                                                                           minMplofLCs_,
                                                                           maxMplofLCs_));

  histograms.h_multiplicity_zminus_numberOfEventsHistogram.push_back(
      ibook.book1D("multiplicity_zminus_numberOfEventsHistogram",
                   "multiplicity numberOfEventsHistogram in z-",
                   nintMplofLCs_,
                   minMplofLCs_,
                   maxMplofLCs_));

  histograms.h_multiplicity_zplus_numberOfEventsHistogram.push_back(
      ibook.book1D("multiplicity_zplus_numberOfEventsHistogram",
                   "multiplicity numberOfEventsHistogram in z+",
                   nintMplofLCs_,
                   minMplofLCs_,
                   maxMplofLCs_));

  histograms.h_multiplicityOfLCinTST_vs_layercluster_zminus.push_back(
      ibook.book2D("multiplicityOfLCinTST_vs_layercluster_zminus",
                   "Multiplicity vs Layer number in z-;LayerCluster multiplicity in Tracksters;layer number",
                   nintMplofLCs_,
                   minMplofLCs_,
                   maxMplofLCs_,
                   layers,
                   0.,
                   (float)layers));

  histograms.h_multiplicityOfLCinTST_vs_layercluster_zplus.push_back(
      ibook.book2D("multiplicityOfLCinTST_vs_layercluster_zplus",
                   "Multiplicity vs Layer number in z+;LayerCluster multiplicity in Tracksters;layer number",
                   nintMplofLCs_,
                   minMplofLCs_,
                   maxMplofLCs_,
                   layers,
                   0.,
                   (float)layers));

  histograms.h_multiplicityOfLCinTST_vs_layerclusterenergy.push_back(
      ibook.book2D("multiplicityOfLCinTST_vs_layerclusterenergy",
                   "Multiplicity vs Layer cluster energy;LayerCluster multiplicity in Tracksters;Cluster energy [GeV]",
                   nintMplofLCs_,
                   minMplofLCs_,
                   maxMplofLCs_,
                   nintClEnepermultiplicity_,
                   minClEnepermultiplicity_,
                   maxClEnepermultiplicity_));

  histograms.h_trackster_pt.push_back(
      ibook.book1D("trackster_pt", "Pt of the Trackster;Trackster p_{T} [GeV]", nintPt_, minPt_, maxPt_));
  histograms.h_trackster_eta.push_back(
      ibook.book1D("trackster_eta", "Eta of the Trackster;Trackster #eta", nintEta_, minEta_, maxEta_));
  histograms.h_trackster_phi.push_back(
      ibook.book1D("trackster_phi", "Phi of the Trackster;Trackster #phi", nintPhi_, minPhi_, maxPhi_));
  histograms.h_trackster_energy.push_back(
      ibook.book1D("trackster_energy", "Energy of the Trackster;Trackster energy [GeV]", nintEne_, minEne_, maxEne_));
  histograms.h_trackster_x.push_back(
      ibook.book1D("trackster_x", "X position of the Trackster;Trackster x", nintX_, minX_, maxX_));
  histograms.h_trackster_y.push_back(
      ibook.book1D("trackster_y", "Y position of the Trackster;Trackster y", nintY_, minY_, maxY_));
  histograms.h_trackster_z.push_back(
      ibook.book1D("trackster_z", "Z position of the Trackster;Trackster z", nintZ_, minZ_, maxZ_));
  histograms.h_trackster_firstlayer.push_back(ibook.book1D(
      "trackster_firstlayer", "First layer of the Trackster;Trackster First Layer", 2 * layers, 0., (float)2 * layers));
  histograms.h_trackster_lastlayer.push_back(ibook.book1D(
      "trackster_lastlayer", "Last layer of the Trackster;Trackster Last Layer", 2 * layers, 0., (float)2 * layers));
  histograms.h_trackster_layersnum.push_back(
      ibook.book1D("trackster_layersnum",
                   "Number of layers of the Trackster;Trackster Number of Layers",
                   2 * layers,
                   0.,
                   (float)2 * layers));
}

void HGVHistoProducerAlgo::bookTracksterSTSHistos(DQMStore::IBooker& ibook,
                                                  Histograms& histograms,
                                                  const validationType valType) {
  const string ref[] = {"caloparticle", "simtrackster", "simtrackster_fromCP"};
  const string refT[] = {"CaloParticle", "SimTrackster", "SimTrackster_fromCP"};
  // Must be in sync with labels in PostProcessorHGCAL_cfi.py
  const string val[] = {"_Link", "_PR"};
  //const string val[] = {"_CP", "_PR", "_Link"};
  const string rtos = ";score Reco-to-Sim";
  const string stor = ";score Sim-to-Reco";
  const string shREnFr = ";shared Reco energy fraction";
  const string shSEnFr = ";shared Sim energy fraction";

  histograms.h_score_trackster2caloparticle[valType].push_back(
      ibook.book1D("Score_trackster2" + ref[valType],
                   "Score of Trackster per " + refT[valType] + rtos,
                   nintScore_,
                   minScore_,
                   maxScore_));
  histograms.h_score_trackster2bestCaloparticle[valType].push_back(
      ibook.book1D("ScoreFake_trackster2" + ref[valType],
                   "Score of Trackster per best " + refT[valType] + rtos,
                   nintScore_,
                   minScore_,
                   maxScore_));
  histograms.h_score_trackster2bestCaloparticle2[valType].push_back(
      ibook.book1D("ScoreMerge_trackster2" + ref[valType],
                   "Score of Trackster per 2^{nd} best " + refT[valType] + rtos,
                   nintScore_,
                   minScore_,
                   maxScore_));
  histograms.h_score_caloparticle2trackster[valType].push_back(
      ibook.book1D("Score_" + ref[valType] + "2trackster",
                   "Score of " + refT[valType] + " per Trackster" + stor,
                   nintScore_,
                   minScore_,
                   maxScore_));
  histograms.h_scorePur_caloparticle2trackster[valType].push_back(
      ibook.book1D("ScorePur_" + ref[valType] + "2trackster",
                   "Score of " + refT[valType] + " per best Trackster" + stor,
                   nintScore_,
                   minScore_,
                   maxScore_));
  histograms.h_scoreDupl_caloparticle2trackster[valType].push_back(
      ibook.book1D("ScoreDupl_" + ref[valType] + "2trackster",
                   "Score of " + refT[valType] + " per 2^{nd} best Trackster" + stor,
                   nintScore_,
                   minScore_,
                   maxScore_));
  histograms.h_energy_vs_score_trackster2caloparticle[valType].push_back(
      ibook.book2D("Energy_vs_Score_trackster2" + refT[valType],
                   "Energy vs Score of Trackster per " + refT[valType] + rtos + shREnFr,
                   nintScore_,
                   minScore_,
                   maxScore_,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_energy_vs_score_trackster2bestCaloparticle[valType].push_back(
      ibook.book2D("Energy_vs_Score_trackster2best" + refT[valType],
                   "Energy vs Score of Trackster per best " + refT[valType] + rtos + shREnFr,
                   nintScore_,
                   minScore_,
                   maxScore_,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_energy_vs_score_trackster2bestCaloparticle2[valType].push_back(
      ibook.book2D("Energy_vs_Score_trackster2secBest" + refT[valType],
                   "Energy vs Score of Trackster per 2^{nd} best " + refT[valType] + rtos + shREnFr,
                   nintScore_,
                   minScore_,
                   maxScore_,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_energy_vs_score_caloparticle2trackster[valType].push_back(
      ibook.book2D("Energy_vs_Score_" + ref[valType] + "2Trackster",
                   "Energy vs Score of " + refT[valType] + " per Trackster" + stor + shSEnFr,
                   nintScore_,
                   minScore_,
                   maxScore_,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_energy_vs_score_caloparticle2bestTrackster[valType].push_back(
      ibook.book2D("Energy_vs_Score_" + ref[valType] + "2bestTrackster",
                   "Energy vs Score of " + refT[valType] + " per best Trackster" + stor + shSEnFr,
                   nintScore_,
                   minScore_,
                   maxScore_,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_energy_vs_score_caloparticle2bestTrackster2[valType].push_back(
      ibook.book2D("Energy_vs_Score_" + ref[valType] + "2secBestTrackster",
                   "Energy vs Score of " + refT[valType] + " per 2^{nd} best Trackster" + stor + shSEnFr,
                   nintScore_,
                   minScore_,
                   maxScore_,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));

  // Back to all Tracksters
  // eta
  histograms.h_num_trackster_eta[valType].push_back(ibook.book1D(
      "Num_Trackster_Eta" + val[valType], "Num Trackster Eta per Trackster;#eta", nintEta_, minEta_, maxEta_));
  histograms.h_numMerge_trackster_eta[valType].push_back(ibook.book1D("NumMerge_Trackster_Eta" + val[valType],
                                                                      "Num Merge Trackster Eta per Trackster;#eta",
                                                                      nintEta_,
                                                                      minEta_,
                                                                      maxEta_));
  histograms.h_denom_trackster_eta[valType].push_back(ibook.book1D(
      "Denom_Trackster_Eta" + val[valType], "Denom Trackster Eta per Trackster;#eta", nintEta_, minEta_, maxEta_));
  // phi
  histograms.h_num_trackster_phi[valType].push_back(ibook.book1D(
      "Num_Trackster_Phi" + val[valType], "Num Trackster Phi per Trackster;#phi", nintPhi_, minPhi_, maxPhi_));
  histograms.h_numMerge_trackster_phi[valType].push_back(ibook.book1D("NumMerge_Trackster_Phi" + val[valType],
                                                                      "Num Merge Trackster Phi per Trackster;#phi",
                                                                      nintPhi_,
                                                                      minPhi_,
                                                                      maxPhi_));
  histograms.h_denom_trackster_phi[valType].push_back(ibook.book1D(
      "Denom_Trackster_Phi" + val[valType], "Denom Trackster Phi per Trackster;#phi", nintPhi_, minPhi_, maxPhi_));
  // energy
  histograms.h_num_trackster_en[valType].push_back(ibook.book1D("Num_Trackster_Energy" + val[valType],
                                                                "Num Trackster Energy per Trackster;energy [GeV]",
                                                                nintEne_,
                                                                minEne_,
                                                                maxEne_));
  histograms.h_numMerge_trackster_en[valType].push_back(
      ibook.book1D("NumMerge_Trackster_Energy" + val[valType],
                   "Num Merge Trackster Energy per Trackster;energy [GeV]",
                   nintEne_,
                   minEne_,
                   maxEne_));
  histograms.h_denom_trackster_en[valType].push_back(ibook.book1D("Denom_Trackster_Energy" + val[valType],
                                                                  "Denom Trackster Energy per Trackster;energy [GeV]",
                                                                  nintEne_,
                                                                  minEne_,
                                                                  maxEne_));
  // pT
  histograms.h_num_trackster_pt[valType].push_back(ibook.book1D(
      "Num_Trackster_Pt" + val[valType], "Num Trackster p_{T} per Trackster;p_{T} [GeV]", nintPt_, minPt_, maxPt_));
  histograms.h_numMerge_trackster_pt[valType].push_back(
      ibook.book1D("NumMerge_Trackster_Pt" + val[valType],
                   "Num Merge Trackster p_{T} per Trackster;p_{T} [GeV]",
                   nintPt_,
                   minPt_,
                   maxPt_));
  histograms.h_denom_trackster_pt[valType].push_back(ibook.book1D(
      "Denom_Trackster_Pt" + val[valType], "Denom Trackster p_{T} per Trackster;p_{T} [GeV]", nintPt_, minPt_, maxPt_));

  histograms.h_sharedenergy_trackster2caloparticle[valType].push_back(
      ibook.book1D("SharedEnergy_trackster2" + ref[valType],
                   "Shared Energy of Trackster per " + refT[valType] + shREnFr,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_trackster2bestCaloparticle[valType].push_back(
      ibook.book1D("SharedEnergy_trackster2" + ref[valType] + "_assoc",
                   "Shared Energy of Trackster per best " + refT[valType] + shREnFr,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_trackster2bestCaloparticle_vs_eta[valType].push_back(
      ibook.bookProfile("SharedEnergy_trackster2" + ref[valType] + "_assoc_vs_eta",
                        "Shared Energy of Trackster vs #eta per best " + refT[valType] + ";Trackster #eta" + shREnFr,
                        nintEta_,
                        minEta_,
                        maxEta_,
                        minTSTSharedEneFrac_,
                        maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_trackster2bestCaloparticle_vs_phi[valType].push_back(
      ibook.bookProfile("SharedEnergy_trackster2" + ref[valType] + "_assoc_vs_phi",
                        "Shared Energy of Trackster vs #phi per best " + refT[valType] + ";Trackster #phi" + shREnFr,
                        nintPhi_,
                        minPhi_,
                        maxPhi_,
                        minTSTSharedEneFrac_,
                        maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_trackster2bestCaloparticle2[valType].push_back(
      ibook.book1D("SharedEnergy_trackster2" + ref[valType] + "_assoc2",
                   "Shared Energy of Trackster per 2^{nd} best " + refT[valType] + shREnFr,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));

  histograms.h_sharedenergy_caloparticle2trackster[valType].push_back(
      ibook.book1D("SharedEnergy_" + ref[valType] + "2trackster",
                   "Shared Energy of " + refT[valType] + " per Trackster" + shSEnFr,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_caloparticle2trackster_assoc[valType].push_back(
      ibook.book1D("SharedEnergy_" + ref[valType] + "2trackster_assoc",
                   "Shared Energy of " + refT[valType] + " per best Trackster" + shSEnFr,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_caloparticle2trackster_assoc_vs_eta[valType].push_back(ibook.bookProfile(
      "SharedEnergy_" + ref[valType] + "2trackster_assoc_vs_eta",
      "Shared Energy of " + refT[valType] + " vs #eta per best Trackster;" + refT[valType] + " #eta" + shSEnFr,
      nintEta_,
      minEta_,
      maxEta_,
      minTSTSharedEneFrac_,
      maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_caloparticle2trackster_assoc_vs_phi[valType].push_back(ibook.bookProfile(
      "SharedEnergy_" + ref[valType] + "2trackster_assoc_vs_phi",
      "Shared Energy of " + refT[valType] + " vs #phi per best Trackster;" + refT[valType] + " #phi" + shSEnFr,
      nintPhi_,
      minPhi_,
      maxPhi_,
      minTSTSharedEneFrac_,
      maxTSTSharedEneFrac_));
  histograms.h_sharedenergy_caloparticle2trackster_assoc2[valType].push_back(
      ibook.book1D("SharedEnergy_" + ref[valType] + "2trackster_assoc2",
                   "Shared Energy of " + refT[valType] + " per 2^{nd} best Trackster;" + shSEnFr,
                   nintSharedEneFrac_,
                   minTSTSharedEneFrac_,
                   maxTSTSharedEneFrac_));

  // eta
  histograms.h_numEff_caloparticle_eta[valType].push_back(
      ibook.book1D("NumEff_" + refT[valType] + "_Eta",
                   "Num Efficiency " + refT[valType] + " Eta per Trackster;#eta",
                   nintEta_,
                   minEta_,
                   maxEta_));
  histograms.h_num_caloparticle_eta[valType].push_back(
      ibook.book1D("Num_" + refT[valType] + "_Eta",
                   "Num Purity " + refT[valType] + " Eta per Trackster;#eta",
                   nintEta_,
                   minEta_,
                   maxEta_));
  histograms.h_numDup_trackster_eta[valType].push_back(ibook.book1D(
      "NumDup_Trackster_Eta" + val[valType], "Num Duplicate Trackster vs Eta;#eta", nintEta_, minEta_, maxEta_));
  histograms.h_denom_caloparticle_eta[valType].push_back(
      ibook.book1D("Denom_" + refT[valType] + "_Eta",
                   "Denom " + refT[valType] + " Eta per Trackster;#eta",
                   nintEta_,
                   minEta_,
                   maxEta_));
  // phi
  histograms.h_numEff_caloparticle_phi[valType].push_back(
      ibook.book1D("NumEff_" + refT[valType] + "_Phi",
                   "Num Efficiency " + refT[valType] + " Phi per Trackster;#phi",
                   nintPhi_,
                   minPhi_,
                   maxPhi_));
  histograms.h_num_caloparticle_phi[valType].push_back(
      ibook.book1D("Num_" + refT[valType] + "_Phi",
                   "Num Purity " + refT[valType] + " Phi per Trackster;#phi",
                   nintPhi_,
                   minPhi_,
                   maxPhi_));
  histograms.h_numDup_trackster_phi[valType].push_back(ibook.book1D(
      "NumDup_Trackster_Phi" + val[valType], "Num Duplicate Trackster vs Phi;#phi", nintPhi_, minPhi_, maxPhi_));
  histograms.h_denom_caloparticle_phi[valType].push_back(
      ibook.book1D("Denom_" + refT[valType] + "_Phi",
                   "Denom " + refT[valType] + " Phi per Trackster;#phi",
                   nintPhi_,
                   minPhi_,
                   maxPhi_));
  // energy
  histograms.h_numEff_caloparticle_en[valType].push_back(
      ibook.book1D("NumEff_" + refT[valType] + "_Energy",
                   "Num Efficiency " + refT[valType] + " Energy per Trackster;energy [GeV]",
                   nintEne_,
                   minEne_,
                   maxEne_));
  histograms.h_num_caloparticle_en[valType].push_back(
      ibook.book1D("Num_" + refT[valType] + "_Energy",
                   "Num Purity " + refT[valType] + " Energy per Trackster;energy [GeV]",
                   nintEne_,
                   minEne_,
                   maxEne_));
  histograms.h_numDup_trackster_en[valType].push_back(ibook.book1D("NumDup_Trackster_Energy" + val[valType],
                                                                   "Num Duplicate Trackster vs Energy;energy [GeV]",
                                                                   nintEne_,
                                                                   minEne_,
                                                                   maxEne_));
  histograms.h_denom_caloparticle_en[valType].push_back(
      ibook.book1D("Denom_" + refT[valType] + "_Energy",
                   "Denom " + refT[valType] + " Energy per Trackster;energy [GeV]",
                   nintEne_,
                   minEne_,
                   maxEne_));
  // pT
  histograms.h_numEff_caloparticle_pt[valType].push_back(
      ibook.book1D("NumEff_" + refT[valType] + "_Pt",
                   "Num Efficiency " + refT[valType] + " p_{T} per Trackster;p_{T} [GeV]",
                   nintPt_,
                   minPt_,
                   maxPt_));
  histograms.h_num_caloparticle_pt[valType].push_back(
      ibook.book1D("Num_" + refT[valType] + "_Pt",
                   "Num Purity " + refT[valType] + " p_{T} per Trackster;p_{T} [GeV]",
                   nintPt_,
                   minPt_,
                   maxPt_));
  histograms.h_numDup_trackster_pt[valType].push_back(ibook.book1D(
      "NumDup_Trackster_Pt" + val[valType], "Num Duplicate Trackster vs p_{T};p_{T} [GeV]", nintPt_, minPt_, maxPt_));
  histograms.h_denom_caloparticle_pt[valType].push_back(
      ibook.book1D("Denom_" + refT[valType] + "_Pt",
                   "Denom " + refT[valType] + " p_{T} per Trackster;p_{T} [GeV]",
                   nintPt_,
                   minPt_,
                   maxPt_));
}

void HGVHistoProducerAlgo::fill_info_histos(const Histograms& histograms, unsigned int layers) const {
  // Save some info straight from geometry to avoid mistakes from updates
  //----------- TODO ----------------------------------------------------------
  // For now values returned for 'lastLayerFHzp': '104', 'lastLayerFHzm': '52' are not the one expected.
  // Will come back to this when there will be info in CMSSW to put in DQM file.
  histograms.lastLayerEEzm->Fill(recHitTools_->lastLayerEE());
  histograms.lastLayerFHzm->Fill(recHitTools_->lastLayerFH());
  histograms.maxlayerzm->Fill(layers);
  histograms.lastLayerEEzp->Fill(recHitTools_->lastLayerEE() + layers);
  histograms.lastLayerFHzp->Fill(recHitTools_->lastLayerFH() + layers);
  histograms.maxlayerzp->Fill(layers + layers);
}

void HGVHistoProducerAlgo::fill_caloparticle_histos(const Histograms& histograms,
                                                    int pdgid,
                                                    const CaloParticle& caloParticle,
                                                    std::vector<SimVertex> const& simVertices,
                                                    unsigned int layers,
                                                    std::unordered_map<DetId, const HGCRecHit*> const& hitMap) const {
  const auto eta = getEta(caloParticle.eta());
  if (histograms.h_caloparticle_eta.count(pdgid)) {
    histograms.h_caloparticle_eta.at(pdgid)->Fill(eta);
  }
  if (histograms.h_caloparticle_eta_Zorigin.count(pdgid)) {
    histograms.h_caloparticle_eta_Zorigin.at(pdgid)->Fill(
        simVertices.at(caloParticle.g4Tracks()[0].vertIndex()).position().z(), eta);
  }

  if (histograms.h_caloparticle_energy.count(pdgid)) {
    histograms.h_caloparticle_energy.at(pdgid)->Fill(caloParticle.energy());
  }
  if (histograms.h_caloparticle_pt.count(pdgid)) {
    histograms.h_caloparticle_pt.at(pdgid)->Fill(caloParticle.pt());
  }
  if (histograms.h_caloparticle_phi.count(pdgid)) {
    histograms.h_caloparticle_phi.at(pdgid)->Fill(caloParticle.phi());
  }

  if (histograms.h_caloparticle_nSimClusters.count(pdgid)) {
    histograms.h_caloparticle_nSimClusters.at(pdgid)->Fill(caloParticle.simClusters().size());

    int simHits = 0;
    int minLayerId = 999;
    int maxLayerId = 0;

    int simHits_matched = 0;
    int minLayerId_matched = 999;
    int maxLayerId_matched = 0;

    float energy = 0.;
    std::map<int, double> totenergy_layer;

    float hitEnergyWeight_invSum = 0;
    std::vector<std::pair<DetId, float>> haf_cp;
    for (const auto& sc : caloParticle.simClusters()) {
      LogDebug("HGCalValidator") << " This sim cluster has " << sc->hits_and_fractions().size() << " simHits and "
                                 << sc->energy() << " energy. " << std::endl;
      simHits += sc->hits_and_fractions().size();
      for (auto const& h_and_f : sc->hits_and_fractions()) {
        const auto hitDetId = h_and_f.first;
        const int layerId =
            recHitTools_->getLayerWithOffset(hitDetId) + layers * ((recHitTools_->zside(hitDetId) + 1) >> 1) - 1;
        // set to 0 if matched RecHit not found
        int layerId_matched_min = 999;
        int layerId_matched_max = 0;
        std::unordered_map<DetId, const HGCRecHit*>::const_iterator itcheck = hitMap.find(hitDetId);
        if (itcheck != hitMap.end()) {
          layerId_matched_min = layerId;
          layerId_matched_max = layerId;
          simHits_matched++;

          const auto hitEn = itcheck->second->energy();
          hitEnergyWeight_invSum += pow(hitEn, 2);
          const auto hitFr = h_and_f.second;
          const auto hitEnFr = hitEn * hitFr;
          energy += hitEnFr;
          histograms.h_caloparticle_nHits_matched_energy.at(pdgid)->Fill(hitEnFr);
          histograms.h_caloparticle_nHits_matched_energy_layer.at(pdgid)->Fill(layerId, hitEnFr);

          if (totenergy_layer.find(layerId) != totenergy_layer.end()) {
            totenergy_layer[layerId] = totenergy_layer.at(layerId) + hitEn;
          } else {
            totenergy_layer.emplace(layerId, hitEn);
          }
          if (caloParticle.simClusters().size() == 1)
            histograms.h_caloparticle_nHits_matched_energy_layer_1SimCl.at(pdgid)->Fill(layerId, hitEnFr);

          auto found = std::find_if(std::begin(haf_cp),
                                    std::end(haf_cp),
                                    [&hitDetId](const std::pair<DetId, float>& v) { return v.first == hitDetId; });
          if (found != haf_cp.end())
            found->second += hitFr;
          else
            haf_cp.emplace_back(hitDetId, hitFr);

        } else {
          LogDebug("HGCalValidator") << "   matched to RecHit NOT found !" << std::endl;
        }

        minLayerId = std::min(minLayerId, layerId);
        maxLayerId = std::max(maxLayerId, layerId);
        minLayerId_matched = std::min(minLayerId_matched, layerId_matched_min);
        maxLayerId_matched = std::max(maxLayerId_matched, layerId_matched_max);
      }
      LogDebug("HGCalValidator") << std::endl;
    }  // End loop over SimClusters of CaloParticle
    if (hitEnergyWeight_invSum)
      hitEnergyWeight_invSum = 1 / hitEnergyWeight_invSum;

    histograms.h_caloparticle_firstlayer.at(pdgid)->Fill(minLayerId);
    histograms.h_caloparticle_lastlayer.at(pdgid)->Fill(maxLayerId);
    histograms.h_caloparticle_layersnum.at(pdgid)->Fill(int(maxLayerId - minLayerId));

    histograms.h_caloparticle_firstlayer_matchedtoRecHit.at(pdgid)->Fill(minLayerId_matched);
    histograms.h_caloparticle_lastlayer_matchedtoRecHit.at(pdgid)->Fill(maxLayerId_matched);
    histograms.h_caloparticle_layersnum_matchedtoRecHit.at(pdgid)->Fill(int(maxLayerId_matched - minLayerId_matched));

    histograms.h_caloparticle_nHitsInSimClusters.at(pdgid)->Fill((float)simHits);
    histograms.h_caloparticle_nHitsInSimClusters_matchedtoRecHit.at(pdgid)->Fill((float)simHits_matched);
    histograms.h_caloparticle_selfenergy.at(pdgid)->Fill((float)energy);
    histograms.h_caloparticle_energyDifference.at(pdgid)->Fill((float)1. - energy / caloParticle.energy());

    //Calculate sum energy per-layer
    auto i = totenergy_layer.begin();
    double sum_energy = 0.0;
    while (i != totenergy_layer.end()) {
      sum_energy += i->second;
      histograms.h_caloparticle_sum_energy_layer.at(pdgid)->Fill(i->first, sum_energy / caloParticle.energy() * 100.);
      i++;
    }

    for (auto const& haf : haf_cp) {
      const auto hitEn = hitMap.find(haf.first)->second->energy();
      const auto weight = pow(hitEn, 2);
      histograms.h_caloparticle_fractions.at(pdgid)->Fill(haf.second, weight * hitEnergyWeight_invSum);
      histograms.h_caloparticle_fractions_weight.at(pdgid)->Fill(haf.second, weight * hitEnergyWeight_invSum, weight);
    }
  }
}

void HGVHistoProducerAlgo::HGVHistoProducerAlgo::fill_simCluster_histos(const Histograms& histograms,
                                                                        std::vector<SimCluster> const& simClusters,
                                                                        unsigned int layers,
                                                                        std::vector<int> thicknesses) const {
  //Each event to be treated as two events: an event in +ve endcap,
  //plus another event in -ve endcap. In this spirit there will be
  //a layer variable (layerid) that maps the layers in :
  //-z: 0->49
  //+z: 50->99

  //To keep track of total num of simClusters per layer
  //tnscpl[layerid]
  std::vector<int> tnscpl(1000, 0);  //tnscpl.clear(); tnscpl.reserve(1000);

  //To keep track of the total num of clusters per thickness in plus and in minus endcaps
  std::map<std::string, int> tnscpthplus;
  tnscpthplus.clear();
  std::map<std::string, int> tnscpthminus;
  tnscpthminus.clear();
  //At the beginning of the event all layers should be initialized to zero total clusters per thickness
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    tnscpthplus.insert(std::pair<std::string, int>(std::to_string(*it), 0));
    tnscpthminus.insert(std::pair<std::string, int>(std::to_string(*it), 0));
  }
  //To keep track of the total num of simClusters with mixed thickness hits per event
  tnscpthplus.insert(std::pair<std::string, int>("mixed", 0));
  tnscpthminus.insert(std::pair<std::string, int>("mixed", 0));

  //loop through simClusters
  for (const auto& sc : simClusters) {
    //Auxillary variables to count the number of different kind of hits in each simCluster
    int nthhits120p = 0;
    int nthhits200p = 0;
    int nthhits300p = 0;
    int nthhitsscintp = 0;
    int nthhits120m = 0;
    int nthhits200m = 0;
    int nthhits300m = 0;
    int nthhitsscintm = 0;
    //For the hits thickness of the layer cluster.
    double thickness = 0.;
    //To keep track if we added the simCluster in a specific layer
    std::vector<int> occurenceSCinlayer(1000, 0);  //[layerid][0 if not added]

    //loop through hits of the simCluster
    for (const auto& hAndF : sc.hits_and_fractions()) {
      const DetId sh_detid = hAndF.first;

      //The layer the cluster belongs to. As mentioned in the mapping above, it takes into account -z and +z.
      int layerid =
          recHitTools_->getLayerWithOffset(sh_detid) + layers * ((recHitTools_->zside(sh_detid) + 1) >> 1) - 1;
      //zside that the current cluster belongs to.
      int zside = recHitTools_->zside(sh_detid);

      //add the simCluster to the relevant layer. A SimCluster may give contribution to several layers.
      if (occurenceSCinlayer[layerid] == 0) {
        tnscpl[layerid]++;
      }
      occurenceSCinlayer[layerid]++;

      if (sh_detid.det() == DetId::Forward || sh_detid.det() == DetId::HGCalEE || sh_detid.det() == DetId::HGCalHSi) {
        thickness = recHitTools_->getSiThickness(sh_detid);
      } else if (sh_detid.det() == DetId::HGCalHSc) {
        thickness = -1;
      } else {
        LogDebug("HGCalValidator") << "These are HGCal simClusters, you shouldn't be here !!! " << layerid << "\n";
        continue;
      }

      if ((thickness == 120.) && (zside > 0.)) {
        nthhits120p++;
      } else if ((thickness == 120.) && (zside < 0.)) {
        nthhits120m++;
      } else if ((thickness == 200.) && (zside > 0.)) {
        nthhits200p++;
      } else if ((thickness == 200.) && (zside < 0.)) {
        nthhits200m++;
      } else if ((thickness == 300.) && (zside > 0.)) {
        nthhits300p++;
      } else if ((thickness == 300.) && (zside < 0.)) {
        nthhits300m++;
      } else if ((thickness == -1) && (zside > 0.)) {
        nthhitsscintp++;
      } else if ((thickness == -1) && (zside < 0.)) {
        nthhitsscintm++;
      } else {  //assert(0);
        LogDebug("HGCalValidator")
            << " You are running a geometry that contains thicknesses different than the normal ones. "
            << "\n";
      }

    }  //end of loop through hits

    //Check for simultaneously having hits of different kind. Checking at least two combinations is sufficient.
    if ((nthhits120p != 0 && nthhits200p != 0) || (nthhits120p != 0 && nthhits300p != 0) ||
        (nthhits120p != 0 && nthhitsscintp != 0) || (nthhits200p != 0 && nthhits300p != 0) ||
        (nthhits200p != 0 && nthhitsscintp != 0) || (nthhits300p != 0 && nthhitsscintp != 0)) {
      tnscpthplus["mixed"]++;
    } else if ((nthhits120p != 0 || nthhits200p != 0 || nthhits300p != 0 || nthhitsscintp != 0)) {
      //This is a cluster with hits of one kind
      tnscpthplus[std::to_string((int)thickness)]++;
    }
    if ((nthhits120m != 0 && nthhits200m != 0) || (nthhits120m != 0 && nthhits300m != 0) ||
        (nthhits120m != 0 && nthhitsscintm != 0) || (nthhits200m != 0 && nthhits300m != 0) ||
        (nthhits200m != 0 && nthhitsscintm != 0) || (nthhits300m != 0 && nthhitsscintm != 0)) {
      tnscpthminus["mixed"]++;
    } else if ((nthhits120m != 0 || nthhits200m != 0 || nthhits300m != 0 || nthhitsscintm != 0)) {
      //This is a cluster with hits of one kind
      tnscpthminus[std::to_string((int)thickness)]++;
    }

  }  //end of loop through SimClusters of the event

  //Per layer : Loop 0->99
  for (unsigned ilayer = 0; ilayer < layers * 2; ++ilayer)
    if (histograms.h_simclusternum_perlayer.count(ilayer))
      histograms.h_simclusternum_perlayer.at(ilayer)->Fill(tnscpl[ilayer]);

  //Per thickness
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    if (histograms.h_simclusternum_perthick.count(*it)) {
      histograms.h_simclusternum_perthick.at(*it)->Fill(tnscpthplus[std::to_string(*it)]);
      histograms.h_simclusternum_perthick.at(*it)->Fill(tnscpthminus[std::to_string(*it)]);
    }
  }
  //Mixed thickness clusters
  histograms.h_mixedhitssimcluster_zplus->Fill(tnscpthplus["mixed"]);
  histograms.h_mixedhitssimcluster_zminus->Fill(tnscpthminus["mixed"]);
}

void HGVHistoProducerAlgo::HGVHistoProducerAlgo::fill_simClusterAssociation_histos(
    const Histograms& histograms,
    const int count,
    edm::Handle<reco::CaloClusterCollection> clusterHandle,
    const reco::CaloClusterCollection& clusters,
    edm::Handle<std::vector<SimCluster>> simClusterHandle,
    std::vector<SimCluster> const& simClusters,
    std::vector<size_t> const& sCIndices,
    const std::vector<float>& mask,
    std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
    unsigned int layers,
    const hgcal::RecoToSimCollectionWithSimClusters& scsInLayerClusterMap,
    const hgcal::SimToRecoCollectionWithSimClusters& lcsInSimClusterMap) const {
  //Each event to be treated as two events: an event in +ve endcap,
  //plus another event in -ve endcap. In this spirit there will be
  //a layer variable (layerid) that maps the layers in :
  //-z: 0->49
  //+z: 50->99

  //Will add some general plots on the specific mask in the future.

  layerClusters_to_SimClusters(histograms,
                               count,
                               clusterHandle,
                               clusters,
                               simClusterHandle,
                               simClusters,
                               sCIndices,
                               mask,
                               hitMap,
                               layers,
                               scsInLayerClusterMap,
                               lcsInSimClusterMap);
}

void HGVHistoProducerAlgo::fill_cluster_histos(const Histograms& histograms,
                                               const int count,
                                               const reco::CaloCluster& cluster) const {
  const auto eta = getEta(cluster.eta());
  histograms.h_cluster_eta[count]->Fill(eta);
}

void HGVHistoProducerAlgo::layerClusters_to_CaloParticles(const Histograms& histograms,
                                                          edm::Handle<reco::CaloClusterCollection> clusterHandle,
                                                          const reco::CaloClusterCollection& clusters,
                                                          edm::Handle<std::vector<CaloParticle>> caloParticleHandle,
                                                          std::vector<CaloParticle> const& cP,
                                                          std::vector<size_t> const& cPIndices,
                                                          std::vector<size_t> const& cPSelectedIndices,
                                                          std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
                                                          unsigned int layers,
                                                          const hgcal::RecoToSimCollection& cpsInLayerClusterMap,
                                                          const hgcal::SimToRecoCollection& cPOnLayerMap) const {
  const auto nLayerClusters = clusters.size();

  std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInCluster>> detIdToCaloParticleId_Map;
  std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInCluster>> detIdToLayerClusterId_Map;

  // The association has to be done in an all-vs-all fashion.
  // For this reason use the full set of CaloParticles, with the only filter on bx
  for (const auto& cpId : cPIndices) {
    for (const auto& simCluster : cP[cpId].simClusters()) {
      for (const auto& it_haf : simCluster->hits_and_fractions()) {
        const DetId hitid = (it_haf.first);
        if (hitMap.find(hitid) != hitMap.end()) {
          if (detIdToCaloParticleId_Map.find(hitid) == detIdToCaloParticleId_Map.end()) {
            detIdToCaloParticleId_Map[hitid] = std::vector<HGVHistoProducerAlgo::detIdInfoInCluster>();
            detIdToCaloParticleId_Map[hitid].emplace_back(
                HGVHistoProducerAlgo::detIdInfoInCluster{cpId, it_haf.second});
          } else {
            auto findHitIt =
                std::find(detIdToCaloParticleId_Map[hitid].begin(),
                          detIdToCaloParticleId_Map[hitid].end(),
                          HGVHistoProducerAlgo::detIdInfoInCluster{
                              cpId, 0.f});  // only the first element is used for the matching (overloaded operator==)
            if (findHitIt != detIdToCaloParticleId_Map[hitid].end())
              findHitIt->fraction += it_haf.second;
            else
              detIdToCaloParticleId_Map[hitid].emplace_back(
                  HGVHistoProducerAlgo::detIdInfoInCluster{cpId, it_haf.second});
          }
        }
      }
    }
  }

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const auto& hits_and_fractions = clusters[lcId].hitsAndFractions();
    const auto numberOfHitsInLC = hits_and_fractions.size();

    // This vector will store, for each hit in the Layercluster, the index of
    // the CaloParticle that contributed the most, in terms of energy, to it.
    // Special values are:
    //
    // -2  --> the reconstruction fraction of the RecHit is 0 (used in the past to monitor Halo Hits)
    // -3  --> same as before with the added condition that no CaloParticle has been linked to this RecHit
    // -1  --> the reco fraction is >0, but no CaloParticle has been linked to it
    // >=0 --> index of the linked CaloParticle
    std::vector<int> hitsToCaloParticleId(numberOfHitsInLC);
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId =
        recHitTools_->getLayerWithOffset(firstHitDetId) + layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;

    // This will store the fraction of the CaloParticle energy shared with the LayerCluster: e_shared/cp_energy
    std::unordered_map<unsigned, float> CPEnergyInLC;

    for (unsigned int iHit = 0; iHit < numberOfHitsInLC; iHit++) {
      const DetId rh_detid = hits_and_fractions[iHit].first;
      const auto rhFraction = hits_and_fractions[iHit].second;

      std::unordered_map<DetId, const HGCRecHit*>::const_iterator itcheck = hitMap.find(rh_detid);
      const HGCRecHit* hit = itcheck->second;

      if (detIdToLayerClusterId_Map.find(rh_detid) == detIdToLayerClusterId_Map.end()) {
        detIdToLayerClusterId_Map[rh_detid] = std::vector<HGVHistoProducerAlgo::detIdInfoInCluster>();
      }
      detIdToLayerClusterId_Map[rh_detid].emplace_back(HGVHistoProducerAlgo::detIdInfoInCluster{lcId, rhFraction});

      const auto& hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

      // if the fraction is zero or the hit does not belong to any calo
      // particle, set the caloparticleId for the hit to -1 this will
      // contribute to the number of noise hits

      // MR Remove the case in which the fraction is 0, since this could be a
      // real hit that has been marked as halo.
      if (rhFraction == 0.) {
        hitsToCaloParticleId[iHit] = -2;
      }
      if (hit_find_in_CP == detIdToCaloParticleId_Map.end()) {
        hitsToCaloParticleId[iHit] -= 1;
      } else {
        auto maxCPEnergyInLC = 0.f;
        auto maxCPId = -1;
        for (auto& h : hit_find_in_CP->second) {
          const auto iCP = h.clusterId;
          CPEnergyInLC[iCP] += h.fraction * hit->energy();
          // Keep track of which CaloParticle contributed the most, in terms
          // of energy, to this specific LayerCluster.
          if (CPEnergyInLC[iCP] > maxCPEnergyInLC) {
            maxCPEnergyInLC = CPEnergyInLC[iCP];
            maxCPId = iCP;
          }
        }
        hitsToCaloParticleId[iHit] = maxCPId;
      }
      histograms.h_cellAssociation_perlayer.at(lcLayerId)->Fill(
          hitsToCaloParticleId[iHit] > 0. ? 0. : hitsToCaloParticleId[iHit]);
    }  // End loop over hits on a LayerCluster

  }  // End of loop over LayerClusters

  // Fill the plots to compute the different metrics linked to
  // reco-level, namely fake-rate an merge-rate. In this loop should *not*
  // restrict only to the selected caloParaticles.
  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
    const int lcLayerId =
        recHitTools_->getLayerWithOffset(firstHitDetId) + layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
    histograms.h_denom_layercl_eta_perlayer.at(lcLayerId)->Fill(clusters[lcId].eta());
    histograms.h_denom_layercl_phi_perlayer.at(lcLayerId)->Fill(clusters[lcId].phi());
    //
    const edm::Ref<reco::CaloClusterCollection> lcRef(clusterHandle, lcId);
    const auto& cpsIt = cpsInLayerClusterMap.find(lcRef);
    if (cpsIt == cpsInLayerClusterMap.end())
      continue;

    const auto lc_en = clusters[lcId].energy();
    const auto& cps = cpsIt->val;
    if (lc_en == 0. && !cps.empty()) {
      for (const auto& cpPair : cps)
        histograms.h_score_layercl2caloparticle_perlayer.at(lcLayerId)->Fill(cpPair.second);
      continue;
    }
    for (const auto& cpPair : cps) {
      LogDebug("HGCalValidator") << "layerCluster Id: \t" << lcId << "\t CP id: \t" << cpPair.first.index()
                                 << "\t score \t" << cpPair.second << std::endl;
      histograms.h_score_layercl2caloparticle_perlayer.at(lcLayerId)->Fill(cpPair.second);
      auto const& cp_linked =
          std::find_if(std::begin(cPOnLayerMap[cpPair.first]),
                       std::end(cPOnLayerMap[cpPair.first]),
                       [&lcRef](const std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>& p) {
                         return p.first == lcRef;
                       });
      if (cp_linked ==
          cPOnLayerMap[cpPair.first].end())  // This should never happen by construction of the association maps
        continue;
      histograms.h_sharedenergy_layercl2caloparticle_perlayer.at(lcLayerId)->Fill(cp_linked->second.first / lc_en,
                                                                                  lc_en);
      histograms.h_energy_vs_score_layercl2caloparticle_perlayer.at(lcLayerId)->Fill(cpPair.second,
                                                                                     cp_linked->second.first / lc_en);
    }
    const auto assoc =
        std::count_if(std::begin(cps), std::end(cps), [](const auto& obj) { return obj.second < ScoreCutLCtoCP_; });
    if (assoc) {
      histograms.h_num_layercl_eta_perlayer.at(lcLayerId)->Fill(clusters[lcId].eta());
      histograms.h_num_layercl_phi_perlayer.at(lcLayerId)->Fill(clusters[lcId].phi());
      if (assoc > 1) {
        histograms.h_numMerge_layercl_eta_perlayer.at(lcLayerId)->Fill(clusters[lcId].eta());
        histograms.h_numMerge_layercl_phi_perlayer.at(lcLayerId)->Fill(clusters[lcId].phi());
      }
      const auto& best = std::min_element(
          std::begin(cps), std::end(cps), [](const auto& obj1, const auto& obj2) { return obj1.second < obj2.second; });
      const auto& best_cp_linked =
          std::find_if(std::begin(cPOnLayerMap[best->first]),
                       std::end(cPOnLayerMap[best->first]),
                       [&lcRef](const std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>& p) {
                         return p.first == lcRef;
                       });
      if (best_cp_linked ==
          cPOnLayerMap[best->first].end())  // This should never happen by construction of the association maps
        continue;
      histograms.h_sharedenergy_layercl2caloparticle_vs_eta_perlayer.at(lcLayerId)->Fill(
          clusters[lcId].eta(), best_cp_linked->second.first / lc_en);
      histograms.h_sharedenergy_layercl2caloparticle_vs_phi_perlayer.at(lcLayerId)->Fill(
          clusters[lcId].phi(), best_cp_linked->second.first / lc_en);
    }
  }  // End of loop over LayerClusters

  // Here Fill the plots to compute the different metrics linked to
  // gen-level, namely efficiency and duplicate. In this loop should restrict
  // only to the selected caloParaticles.
  for (const auto& cpId : cPSelectedIndices) {
    const edm::Ref<CaloParticleCollection> cpRef(caloParticleHandle, cpId);
    const auto& lcsIt = cPOnLayerMap.find(cpRef);

    std::map<unsigned int, float> cPEnergyOnLayer;
    for (unsigned int layerId = 0; layerId < layers * 2; ++layerId)
      cPEnergyOnLayer[layerId] = 0;

    for (const auto& simCluster : cP[cpId].simClusters()) {
      for (const auto& it_haf : simCluster->hits_and_fractions()) {
        const DetId hitid = (it_haf.first);
        const auto hitLayerId =
            recHitTools_->getLayerWithOffset(hitid) + layers * ((recHitTools_->zside(hitid) + 1) >> 1) - 1;
        std::unordered_map<DetId, const HGCRecHit*>::const_iterator itcheck = hitMap.find(hitid);
        if (itcheck != hitMap.end()) {
          const HGCRecHit* hit = itcheck->second;
          cPEnergyOnLayer[hitLayerId] += it_haf.second * hit->energy();
        }
      }
    }

    for (unsigned int layerId = 0; layerId < layers * 2; ++layerId) {
      if (!cPEnergyOnLayer[layerId])
        continue;

      histograms.h_denom_caloparticle_eta_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().eta());
      histograms.h_denom_caloparticle_phi_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().phi());

      if (lcsIt == cPOnLayerMap.end())
        continue;
      const auto& lcs = lcsIt->val;

      auto getLCLayerId = [&](const unsigned int lcId) {
        const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
        const auto lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId) +
                               layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
        return lcLayerId;
      };

      for (const auto& lcPair : lcs) {
        if (getLCLayerId(lcPair.first.index()) != layerId)
          continue;
        histograms.h_score_caloparticle2layercl_perlayer.at(layerId)->Fill(lcPair.second.second);
        histograms.h_sharedenergy_caloparticle2layercl_perlayer.at(layerId)->Fill(
            lcPair.second.first / cPEnergyOnLayer[layerId], cPEnergyOnLayer[layerId]);
        histograms.h_energy_vs_score_caloparticle2layercl_perlayer.at(layerId)->Fill(
            lcPair.second.second, lcPair.second.first / cPEnergyOnLayer[layerId]);
      }
      const auto assoc = std::count_if(std::begin(lcs), std::end(lcs), [&](const auto& obj) {
        if (getLCLayerId(obj.first.index()) != layerId)
          return false;
        else
          return obj.second.second < ScoreCutCPtoLC_;
      });
      if (assoc) {
        histograms.h_num_caloparticle_eta_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().eta());
        histograms.h_num_caloparticle_phi_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().phi());
        if (assoc > 1) {
          histograms.h_numDup_caloparticle_eta_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().eta());
          histograms.h_numDup_caloparticle_phi_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().phi());
        }
        const auto best = std::min_element(std::begin(lcs), std::end(lcs), [&](const auto& obj1, const auto& obj2) {
          if (getLCLayerId(obj1.first.index()) != layerId)
            return false;
          else if (getLCLayerId(obj2.first.index()) == layerId)
            return obj1.second.second < obj2.second.second;
          else
            return true;
        });
        histograms.h_sharedenergy_caloparticle2layercl_vs_eta_perlayer.at(layerId)->Fill(
            cP[cpId].g4Tracks()[0].momentum().eta(), best->second.first / cPEnergyOnLayer[layerId]);
        histograms.h_sharedenergy_caloparticle2layercl_vs_phi_perlayer.at(layerId)->Fill(
            cP[cpId].g4Tracks()[0].momentum().phi(), best->second.first / cPEnergyOnLayer[layerId]);
      }
    }
  }
}

void HGVHistoProducerAlgo::layerClusters_to_SimClusters(
    const Histograms& histograms,
    const int count,
    edm::Handle<reco::CaloClusterCollection> clusterHandle,
    const reco::CaloClusterCollection& clusters,
    edm::Handle<std::vector<SimCluster>> simClusterHandle,
    std::vector<SimCluster> const& sC,
    std::vector<size_t> const& sCIndices,
    const std::vector<float>& mask,
    std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
    unsigned int layers,
    const hgcal::RecoToSimCollectionWithSimClusters& scsInLayerClusterMap,
    const hgcal::SimToRecoCollectionWithSimClusters& lcsInSimClusterMap) const {
  // Here fill the plots to compute the different metrics linked to
  // reco-level, namely fake-rate and merge-rate. In this loop should *not*
  // restrict only to the selected SimClusters.
  for (unsigned int lcId = 0; lcId < clusters.size(); ++lcId) {
    if (mask[lcId] != 0.) {
      LogDebug("HGCalValidator") << "Skipping layer cluster " << lcId << " not belonging to mask" << std::endl;
      continue;
    }
    const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
    const auto lcLayerId =
        recHitTools_->getLayerWithOffset(firstHitDetId) + layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
    //Although the ones below are already created in the LC to CP association, will
    //recreate them here since in the post processor it looks in a specific directory.
    histograms.h_denom_layercl_in_simcl_eta_perlayer[count].at(lcLayerId)->Fill(clusters[lcId].eta());
    histograms.h_denom_layercl_in_simcl_phi_perlayer[count].at(lcLayerId)->Fill(clusters[lcId].phi());
    //
    const edm::Ref<reco::CaloClusterCollection> lcRef(clusterHandle, lcId);
    const auto& scsIt = scsInLayerClusterMap.find(lcRef);
    if (scsIt == scsInLayerClusterMap.end())
      continue;

    const auto lc_en = clusters[lcId].energy();
    const auto& scs = scsIt->val;
    // If a reconstructed LayerCluster has energy 0 but is linked to at least a
    // SimCluster, then his score should be 1 as set in the associator
    if (lc_en == 0. && !scs.empty()) {
      for (const auto& scPair : scs) {
        histograms.h_score_layercl2simcluster_perlayer[count].at(lcLayerId)->Fill(scPair.second);
      }
      continue;
    }
    //Loop through all SimClusters linked to the layer cluster under study
    for (const auto& scPair : scs) {
      LogDebug("HGCalValidator") << "layerCluster Id: \t" << lcId << "\t SC id: \t" << scPair.first.index()
                                 << "\t score \t" << scPair.second << std::endl;
      //This should be filled #layerClusters in layer x #linked SimClusters
      histograms.h_score_layercl2simcluster_perlayer[count].at(lcLayerId)->Fill(scPair.second);
      auto const& sc_linked =
          std::find_if(std::begin(lcsInSimClusterMap[scPair.first]),
                       std::end(lcsInSimClusterMap[scPair.first]),
                       [&lcRef](const std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>& p) {
                         return p.first == lcRef;
                       });
      if (sc_linked ==
          lcsInSimClusterMap[scPair.first].end())  // This should never happen by construction of the association maps
        continue;
      histograms.h_sharedenergy_layercl2simcluster_perlayer[count].at(lcLayerId)->Fill(sc_linked->second.first / lc_en,
                                                                                       lc_en);
      histograms.h_energy_vs_score_layercl2simcluster_perlayer[count].at(lcLayerId)->Fill(
          scPair.second, sc_linked->second.first / lc_en);
    }
    //Here he counts how many of the linked SimClusters of the layer cluster under study have a score above a certain value.
    const auto assoc =
        std::count_if(std::begin(scs), std::end(scs), [](const auto& obj) { return obj.second < ScoreCutLCtoSC_; });
    if (assoc) {
      histograms.h_num_layercl_in_simcl_eta_perlayer[count].at(lcLayerId)->Fill(clusters[lcId].eta());
      histograms.h_num_layercl_in_simcl_phi_perlayer[count].at(lcLayerId)->Fill(clusters[lcId].phi());
      if (assoc > 1) {
        histograms.h_numMerge_layercl_in_simcl_eta_perlayer[count].at(lcLayerId)->Fill(clusters[lcId].eta());
        histograms.h_numMerge_layercl_in_simcl_phi_perlayer[count].at(lcLayerId)->Fill(clusters[lcId].phi());
      }
      const auto& best = std::min_element(
          std::begin(scs), std::end(scs), [](const auto& obj1, const auto& obj2) { return obj1.second < obj2.second; });
      //From all SimClusters he founds the one with the best (lowest) score and takes his scId
      const auto& best_sc_linked =
          std::find_if(std::begin(lcsInSimClusterMap[best->first]),
                       std::end(lcsInSimClusterMap[best->first]),
                       [&lcRef](const std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>& p) {
                         return p.first == lcRef;
                       });
      if (best_sc_linked ==
          lcsInSimClusterMap[best->first].end())  // This should never happen by construction of the association maps
        continue;
      histograms.h_sharedenergy_layercl2simcluster_vs_eta_perlayer[count].at(lcLayerId)->Fill(
          clusters[lcId].eta(), best_sc_linked->second.first / lc_en);
      histograms.h_sharedenergy_layercl2simcluster_vs_phi_perlayer[count].at(lcLayerId)->Fill(
          clusters[lcId].phi(), best_sc_linked->second.first / lc_en);
    }
  }  // End of loop over LayerClusters

  // Fill the plots to compute the different metrics linked to
  // gen-level, namely efficiency and duplicate. In this loop should restrict
  // only to the selected SimClusters.
  for (const auto& scId : sCIndices) {
    const edm::Ref<SimClusterCollection> scRef(simClusterHandle, scId);
    const auto& lcsIt = lcsInSimClusterMap.find(scRef);

    std::map<unsigned int, float> sCEnergyOnLayer;
    for (unsigned int layerId = 0; layerId < layers * 2; ++layerId)
      sCEnergyOnLayer[layerId] = 0;

    for (const auto& it_haf : sC[scId].hits_and_fractions()) {
      const DetId hitid = (it_haf.first);
      const auto scLayerId =
          recHitTools_->getLayerWithOffset(hitid) + layers * ((recHitTools_->zside(hitid) + 1) >> 1) - 1;
      std::unordered_map<DetId, const HGCRecHit*>::const_iterator itcheck = hitMap.find(hitid);
      if (itcheck != hitMap.end()) {
        const HGCRecHit* hit = itcheck->second;
        sCEnergyOnLayer[scLayerId] += it_haf.second * hit->energy();
      }
    }

    for (unsigned int layerId = 0; layerId < layers * 2; ++layerId) {
      if (!sCEnergyOnLayer[layerId])
        continue;

      histograms.h_denom_simcluster_eta_perlayer[count].at(layerId)->Fill(sC[scId].eta());
      histograms.h_denom_simcluster_phi_perlayer[count].at(layerId)->Fill(sC[scId].phi());

      if (lcsIt == lcsInSimClusterMap.end())
        continue;
      const auto& lcs = lcsIt->val;

      auto getLCLayerId = [&](const unsigned int lcId) {
        const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
        const unsigned int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId) +
                                       layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
        return lcLayerId;
      };

      //Loop through layer clusters linked to the SimCluster under study
      for (const auto& lcPair : lcs) {
        auto lcId = lcPair.first.index();
        if (mask[lcId] != 0.) {
          LogDebug("HGCalValidator") << "Skipping layer cluster " << lcId << " not belonging to mask" << std::endl;
          continue;
        }

        if (getLCLayerId(lcId) != layerId)
          continue;
        histograms.h_score_simcluster2layercl_perlayer[count].at(layerId)->Fill(lcPair.second.second);
        histograms.h_sharedenergy_simcluster2layercl_perlayer[count].at(layerId)->Fill(
            lcPair.second.first / sCEnergyOnLayer[layerId], sCEnergyOnLayer[layerId]);
        histograms.h_energy_vs_score_simcluster2layercl_perlayer[count].at(layerId)->Fill(
            lcPair.second.second, lcPair.second.first / sCEnergyOnLayer[layerId]);
      }
      const auto assoc = std::count_if(std::begin(lcs), std::end(lcs), [&](const auto& obj) {
        if (getLCLayerId(obj.first.index()) != layerId)
          return false;
        else
          return obj.second.second < ScoreCutSCtoLC_;
      });
      if (assoc) {
        histograms.h_num_simcluster_eta_perlayer[count].at(layerId)->Fill(sC[scId].eta());
        histograms.h_num_simcluster_phi_perlayer[count].at(layerId)->Fill(sC[scId].phi());
        if (assoc > 1) {
          histograms.h_numDup_simcluster_eta_perlayer[count].at(layerId)->Fill(sC[scId].eta());
          histograms.h_numDup_simcluster_phi_perlayer[count].at(layerId)->Fill(sC[scId].phi());
        }
        const auto best = std::min_element(std::begin(lcs), std::end(lcs), [&](const auto& obj1, const auto& obj2) {
          if (getLCLayerId(obj1.first.index()) != layerId)
            return false;
          else if (getLCLayerId(obj2.first.index()) == layerId)
            return obj1.second.second < obj2.second.second;
          else
            return true;
        });
        histograms.h_sharedenergy_simcluster2layercl_vs_eta_perlayer[count].at(layerId)->Fill(
            sC[scId].eta(), best->second.first / sCEnergyOnLayer[layerId]);
        histograms.h_sharedenergy_simcluster2layercl_vs_phi_perlayer[count].at(layerId)->Fill(
            sC[scId].phi(), best->second.first / sCEnergyOnLayer[layerId]);
      }
    }
  }
}

void HGVHistoProducerAlgo::fill_generic_cluster_histos(const Histograms& histograms,
                                                       const int count,
                                                       edm::Handle<reco::CaloClusterCollection> clusterHandle,
                                                       const reco::CaloClusterCollection& clusters,
                                                       const Density& densities,
                                                       edm::Handle<std::vector<CaloParticle>> caloParticleHandle,
                                                       std::vector<CaloParticle> const& cP,
                                                       std::vector<size_t> const& cPIndices,
                                                       std::vector<size_t> const& cPSelectedIndices,
                                                       std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
                                                       std::map<double, double> cummatbudg,
                                                       unsigned int layers,
                                                       std::vector<int> thicknesses,
                                                       const hgcal::RecoToSimCollection& cpsInLayerClusterMap,
                                                       const hgcal::SimToRecoCollection& cPOnLayerMap) const {
  //Each event to be treated as two events: an event in +ve endcap,
  //plus another event in -ve endcap. In this spirit there will be
  //a layer variable (layerid) that maps the layers in :
  //-z: 0->51
  //+z: 52->103

  //To keep track of total num of layer clusters per layer
  //tnlcpl[layerid]
  std::vector<int> tnlcpl(1000, 0);  //tnlcpl.clear(); tnlcpl.reserve(1000);

  //To keep track of the total num of clusters per thickness in plus and in minus endcaps
  std::map<std::string, int> tnlcpthplus;
  tnlcpthplus.clear();
  std::map<std::string, int> tnlcpthminus;
  tnlcpthminus.clear();
  //At the beginning of the event all layers should be initialized to zero total clusters per thickness
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    tnlcpthplus.insert(std::pair<std::string, int>(std::to_string(*it), 0));
    tnlcpthminus.insert(std::pair<std::string, int>(std::to_string(*it), 0));
  }
  //To keep track of the total num of clusters with mixed thickness hits per event
  tnlcpthplus.insert(std::pair<std::string, int>("mixed", 0));
  tnlcpthminus.insert(std::pair<std::string, int>("mixed", 0));

  layerClusters_to_CaloParticles(histograms,
                                 clusterHandle,
                                 clusters,
                                 caloParticleHandle,
                                 cP,
                                 cPIndices,
                                 cPSelectedIndices,
                                 hitMap,
                                 layers,
                                 cpsInLayerClusterMap,
                                 cPOnLayerMap);

  //To find out the total amount of energy clustered per layer
  //Initialize with zeros because I see clear gives weird numbers.
  std::vector<double> tecpl(1000, 0.0);  //tecpl.clear(); tecpl.reserve(1000);
  //for the longitudinal depth barycenter
  std::vector<double> ldbar(1000, 0.0);  //ldbar.clear(); ldbar.reserve(1000);

  // Need to compare with the total amount of energy coming from CaloParticles
  double caloparteneplus = 0.;
  double caloparteneminus = 0.;
  for (const auto& cpId : cPIndices) {
    if (cP[cpId].eta() >= 0.) {
      caloparteneplus = caloparteneplus + cP[cpId].energy();
    } else if (cP[cpId].eta() < 0.) {
      caloparteneminus = caloparteneminus + cP[cpId].energy();
    }
  }

  // loop through clusters of the event
  for (const auto& lcId : clusters) {
    const auto seedid = lcId.seed();
    const double seedx = recHitTools_->getPosition(seedid).x();
    const double seedy = recHitTools_->getPosition(seedid).y();
    DetId maxid = findmaxhit(lcId, hitMap);

    // const DetId maxid = lcId.max();
    double maxx = recHitTools_->getPosition(maxid).x();
    double maxy = recHitTools_->getPosition(maxid).y();

    //Auxillary variables to count the number of different kind of hits in each cluster
    int nthhits120p = 0;
    int nthhits200p = 0;
    int nthhits300p = 0;
    int nthhitsscintp = 0;
    int nthhits120m = 0;
    int nthhits200m = 0;
    int nthhits300m = 0;
    int nthhitsscintm = 0;
    //For the hits thickness of the layer cluster.
    double thickness = 0.;
    //The layer the cluster belongs to. As mentioned in the mapping above, it takes into account -z and +z.
    int layerid = 0;
    // Need another layer variable for the longitudinal material budget file reading.
    //In this case need no distinction between -z and +z.
    int lay = 0;
    // Need to save the combination thick_lay
    std::string istr = "";
    //boolean to check for the layer that the cluster belong to. Maybe later will check all the layer hits.
    bool cluslay = true;
    //zside that the current cluster belongs to.
    int zside = 0;

    const auto& hits_and_fractions = lcId.hitsAndFractions();
    for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin();
         it_haf != hits_and_fractions.end();
         ++it_haf) {
      const DetId rh_detid = it_haf->first;
      //The layer that the current hit belongs to
      layerid = recHitTools_->getLayerWithOffset(rh_detid) + layers * ((recHitTools_->zside(rh_detid) + 1) >> 1) - 1;
      lay = recHitTools_->getLayerWithOffset(rh_detid);
      zside = recHitTools_->zside(rh_detid);
      if (rh_detid.det() == DetId::Forward || rh_detid.det() == DetId::HGCalEE || rh_detid.det() == DetId::HGCalHSi) {
        thickness = recHitTools_->getSiThickness(rh_detid);
      } else if (rh_detid.det() == DetId::HGCalHSc) {
        thickness = -1;
      } else {
        LogDebug("HGCalValidator") << "These are HGCal layer clusters, you shouldn't be here !!! " << layerid << "\n";
        continue;
      }

      //Count here only once the layer cluster and save the combination thick_layerid
      std::string curistr = std::to_string((int)thickness);
      std::string lay_string = std::to_string(layerid);
      while (lay_string.size() < 2)
        lay_string.insert(0, "0");
      curistr += "_" + lay_string;
      if (cluslay) {
        tnlcpl[layerid]++;
        istr = curistr;
        cluslay = false;
      }

      if ((thickness == 120.) && (recHitTools_->zside(rh_detid) > 0.)) {
        nthhits120p++;
      } else if ((thickness == 120.) && (recHitTools_->zside(rh_detid) < 0.)) {
        nthhits120m++;
      } else if ((thickness == 200.) && (recHitTools_->zside(rh_detid) > 0.)) {
        nthhits200p++;
      } else if ((thickness == 200.) && (recHitTools_->zside(rh_detid) < 0.)) {
        nthhits200m++;
      } else if ((thickness == 300.) && (recHitTools_->zside(rh_detid) > 0.)) {
        nthhits300p++;
      } else if ((thickness == 300.) && (recHitTools_->zside(rh_detid) < 0.)) {
        nthhits300m++;
      } else if ((thickness == -1) && (recHitTools_->zside(rh_detid) > 0.)) {
        nthhitsscintp++;
      } else if ((thickness == -1) && (recHitTools_->zside(rh_detid) < 0.)) {
        nthhitsscintm++;
      } else {  //assert(0);
        LogDebug("HGCalValidator")
            << " You are running a geometry that contains thicknesses different than the normal ones. "
            << "\n";
      }

      std::unordered_map<DetId, const HGCRecHit*>::const_iterator itcheck = hitMap.find(rh_detid);
      if (itcheck == hitMap.end()) {
        std::ostringstream st1;
        if ((rh_detid.det() == DetId::HGCalEE) || (rh_detid.det() == DetId::HGCalHSi)) {
          st1 << HGCSiliconDetId(rh_detid);
        } else if (rh_detid.det() == DetId::HGCalHSc) {
          st1 << HGCScintillatorDetId(rh_detid);
        } else {
          st1 << HFNoseDetId(rh_detid);
        }
        LogDebug("HGCalValidator") << " You shouldn't be here - Unable to find a hit " << rh_detid.rawId() << " "
                                   << rh_detid.det() << " " << st1.str() << "\n";
        continue;
      }

      const HGCRecHit* hit = itcheck->second;

      //Here for the per cell plots
      //----
      const double hit_x = recHitTools_->getPosition(rh_detid).x();
      const double hit_y = recHitTools_->getPosition(rh_detid).y();
      double distancetoseed = distance(seedx, seedy, hit_x, hit_y);
      double distancetomax = distance(maxx, maxy, hit_x, hit_y);
      if (distancetoseed != 0. && histograms.h_distancetoseedcell_perthickperlayer.count(curistr)) {
        histograms.h_distancetoseedcell_perthickperlayer.at(curistr)->Fill(distancetoseed);
      }
      //----
      if (distancetoseed != 0. && histograms.h_distancetoseedcell_perthickperlayer_eneweighted.count(curistr)) {
        histograms.h_distancetoseedcell_perthickperlayer_eneweighted.at(curistr)->Fill(distancetoseed, hit->energy());
      }
      //----
      if (distancetomax != 0. && histograms.h_distancetomaxcell_perthickperlayer.count(curistr)) {
        histograms.h_distancetomaxcell_perthickperlayer.at(curistr)->Fill(distancetomax);
      }
      //----
      if (distancetomax != 0. && histograms.h_distancetomaxcell_perthickperlayer_eneweighted.count(curistr)) {
        histograms.h_distancetomaxcell_perthickperlayer_eneweighted.at(curistr)->Fill(distancetomax, hit->energy());
      }

      //Let's check the density
      std::map<DetId, float>::const_iterator dit = densities.find(rh_detid);
      if (dit == densities.end()) {
        std::ostringstream st1;
        if ((rh_detid.det() == DetId::HGCalEE) || (rh_detid.det() == DetId::HGCalHSi)) {
          st1 << HGCSiliconDetId(rh_detid);
        } else if (rh_detid.det() == DetId::HGCalHSc) {
          st1 << HGCScintillatorDetId(rh_detid);
        } else {
          st1 << HFNoseDetId(rh_detid);
        }
        LogDebug("HGCalValidator") << " You shouldn't be here - Unable to find a density " << rh_detid.rawId() << " "
                                   << rh_detid.det() << " " << st1.str() << "\n";
        continue;
      }

      if (histograms.h_cellsenedens_perthick.count((int)thickness)) {
        histograms.h_cellsenedens_perthick.at((int)thickness)->Fill(dit->second);
      }

    }  // end of loop through hits and fractions

    //Check for simultaneously having hits of different kind. Checking at least two combinations is sufficient.
    if ((nthhits120p != 0 && nthhits200p != 0) || (nthhits120p != 0 && nthhits300p != 0) ||
        (nthhits120p != 0 && nthhitsscintp != 0) || (nthhits200p != 0 && nthhits300p != 0) ||
        (nthhits200p != 0 && nthhitsscintp != 0) || (nthhits300p != 0 && nthhitsscintp != 0)) {
      tnlcpthplus["mixed"]++;
    } else if ((nthhits120p != 0 || nthhits200p != 0 || nthhits300p != 0 || nthhitsscintp != 0)) {
      //This is a cluster with hits of one kind
      tnlcpthplus[std::to_string((int)thickness)]++;
    }
    if ((nthhits120m != 0 && nthhits200m != 0) || (nthhits120m != 0 && nthhits300m != 0) ||
        (nthhits120m != 0 && nthhitsscintm != 0) || (nthhits200m != 0 && nthhits300m != 0) ||
        (nthhits200m != 0 && nthhitsscintm != 0) || (nthhits300m != 0 && nthhitsscintm != 0)) {
      tnlcpthminus["mixed"]++;
    } else if ((nthhits120m != 0 || nthhits200m != 0 || nthhits300m != 0 || nthhitsscintm != 0)) {
      //This is a cluster with hits of one kind
      tnlcpthminus[std::to_string((int)thickness)]++;
    }

    //To find the thickness with the biggest amount of cells
    std::vector<int> bigamoth;
    bigamoth.clear();
    if (zside > 0) {
      bigamoth.push_back(nthhits120p);
      bigamoth.push_back(nthhits200p);
      bigamoth.push_back(nthhits300p);
      bigamoth.push_back(nthhitsscintp);
    } else if (zside < 0) {
      bigamoth.push_back(nthhits120m);
      bigamoth.push_back(nthhits200m);
      bigamoth.push_back(nthhits300m);
      bigamoth.push_back(nthhitsscintm);
    }
    auto bgth = std::max_element(bigamoth.begin(), bigamoth.end());
    istr = std::to_string(thicknesses[std::distance(bigamoth.begin(), bgth)]);
    std::string lay_string = std::to_string(layerid);
    while (lay_string.size() < 2)
      lay_string.insert(0, "0");
    istr += "_" + lay_string;

    //Here for the per cluster plots that need the thickness_layer info
    if (histograms.h_cellsnum_perthickperlayer.count(istr)) {
      histograms.h_cellsnum_perthickperlayer.at(istr)->Fill(hits_and_fractions.size());
    }

    //Now, with the distance between seed and max cell.
    double distancebetseedandmax = distance(seedx, seedy, maxx, maxy);
    //The thickness_layer combination in this case will use the thickness of the seed as a convention.
    std::string seedstr = std::to_string((int)recHitTools_->getSiThickness(seedid)) + "_" + std::to_string(layerid);
    seedstr += "_" + lay_string;
    if (histograms.h_distancebetseedandmaxcell_perthickperlayer.count(seedstr)) {
      histograms.h_distancebetseedandmaxcell_perthickperlayer.at(seedstr)->Fill(distancebetseedandmax);
    }
    const auto lc_en = lcId.energy();
    if (histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer.count(seedstr)) {
      histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer.at(seedstr)->Fill(distancebetseedandmax,
                                                                                               lc_en);
    }

    //Energy clustered per layer
    tecpl[layerid] = tecpl[layerid] + lc_en;
    ldbar[layerid] = ldbar[layerid] + lc_en * cummatbudg[(double)lay];

  }  //end of loop through clusters of the event

  // First a couple of variables to keep the sum of the energy of all clusters
  double sumeneallcluspl = 0.;
  double sumeneallclusmi = 0.;
  // and the longitudinal variable
  double sumldbarpl = 0.;
  double sumldbarmi = 0.;
  //Per layer : Loop 0->103
  for (unsigned ilayer = 0; ilayer < layers * 2; ++ilayer) {
    if (histograms.h_clusternum_perlayer.count(ilayer)) {
      histograms.h_clusternum_perlayer.at(ilayer)->Fill(tnlcpl[ilayer]);
    }
    // Two times one for plus and one for minus
    //First with the -z endcap
    if (ilayer < layers) {
      if (histograms.h_energyclustered_perlayer.count(ilayer)) {
        if (caloparteneminus != 0.) {
          histograms.h_energyclustered_perlayer.at(ilayer)->Fill(100. * tecpl[ilayer] / caloparteneminus);
        }
      }
      //Keep here the total energy for the event in -z
      sumeneallclusmi = sumeneallclusmi + tecpl[ilayer];
      //And for the longitudinal variable
      sumldbarmi = sumldbarmi + ldbar[ilayer];
    } else {  //Then for the +z
      if (histograms.h_energyclustered_perlayer.count(ilayer)) {
        if (caloparteneplus != 0.) {
          histograms.h_energyclustered_perlayer.at(ilayer)->Fill(100. * tecpl[ilayer] / caloparteneplus);
        }
      }
      //Keep here the total energy for the event in -z
      sumeneallcluspl = sumeneallcluspl + tecpl[ilayer];
      //And for the longitudinal variable
      sumldbarpl = sumldbarpl + ldbar[ilayer];
    }  //end of +z loop

  }  //end of loop over layers

  //Per thickness
  for (std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    if (histograms.h_clusternum_perthick.count(*it)) {
      histograms.h_clusternum_perthick.at(*it)->Fill(tnlcpthplus[std::to_string(*it)]);
      histograms.h_clusternum_perthick.at(*it)->Fill(tnlcpthminus[std::to_string(*it)]);
    }
  }
  //Mixed thickness clusters
  histograms.h_mixedhitscluster_zplus[count]->Fill(tnlcpthplus["mixed"]);
  histograms.h_mixedhitscluster_zminus[count]->Fill(tnlcpthminus["mixed"]);

  //Total energy clustered from all layer clusters (fraction)
  if (caloparteneplus != 0.) {
    histograms.h_energyclustered_zplus[count]->Fill(100. * sumeneallcluspl / caloparteneplus);
  }
  if (caloparteneminus != 0.) {
    histograms.h_energyclustered_zminus[count]->Fill(100. * sumeneallclusmi / caloparteneminus);
  }

  //For the longitudinal depth barycenter
  histograms.h_longdepthbarycentre_zplus[count]->Fill(sumldbarpl / sumeneallcluspl);
  histograms.h_longdepthbarycentre_zminus[count]->Fill(sumldbarmi / sumeneallclusmi);
}

void HGVHistoProducerAlgo::tracksters_to_SimTracksters(
    const Histograms& histograms,
    const int count,
    const ticl::TracksterCollection& tracksters,
    const reco::CaloClusterCollection& layerClusters,
    const ticl::TracksterCollection& simTSs,
    const validationType valType,
    const ticl::TracksterCollection& simTSs_fromCP,
    const std::map<unsigned int, std::vector<unsigned int>>& cpToSc_SimTrackstersMap,
    std::vector<SimCluster> const& sC,
    const edm::ProductID& cPHandle_id,
    std::vector<CaloParticle> const& cP,
    std::vector<size_t> const& cPIndices,
    std::vector<size_t> const& cPSelectedIndices,
    std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
    unsigned int layers) const {
  const auto nTracksters = tracksters.size();
  const auto nSimTracksters = simTSs.size();

  std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInCluster>> detIdSimTSId_Map;
  std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInTrackster>> detIdToTracksterId_Map;
  std::vector<int> tracksters_FakeMerge(nTracksters, 0);
  std::vector<int> tracksters_PurityDuplicate(nTracksters, 0);

  // This vector contains the ids of the SimTracksters contributing with at least one hit to the Trackster and the reconstruction error
  //stsInTrackster[trackster][STSids]
  //Connects a Trackster with all related SimTracksters.
  std::vector<std::vector<std::pair<unsigned int, float>>> stsInTrackster;
  stsInTrackster.resize(nTracksters);

  // cPOnLayer[caloparticle]:
  //1. the sum of all rechits energy times fraction of the relevant simhit related to that calo particle.
  //2. the hits and fractions of that calo particle.
  //3. the layer clusters with matched rechit id.
  std::unordered_map<int, caloParticleOnLayer> cPOnLayer;
  std::unordered_map<int, std::vector<caloParticleOnLayer>> sCOnLayer;
  //Consider CaloParticles coming from the hard scatterer, excluding the PU contribution.
  for (const auto cpIndex : cPIndices) {
    cPOnLayer[cpIndex].caloParticleId = cpIndex;
    cPOnLayer[cpIndex].energy = 0.f;
    cPOnLayer[cpIndex].hits_and_fractions.clear();
    const auto nSC_inCP = sC.size();
    sCOnLayer[cpIndex].resize(nSC_inCP);
    for (unsigned int iSC = 0; iSC < nSC_inCP; iSC++) {
      sCOnLayer[cpIndex][iSC].caloParticleId = cpIndex;
      sCOnLayer[cpIndex][iSC].energy = 0.f;
      sCOnLayer[cpIndex][iSC].hits_and_fractions.clear();
    }
  }

  auto getCPId = [](const ticl::Trackster& simTS,
                    const unsigned int iSTS,
                    const edm::ProductID& cPHandle_id,
                    const std::map<unsigned int, std::vector<unsigned int>>& cpToSc_SimTrackstersMap,
                    const ticl::TracksterCollection& simTSs_fromCP) {
    unsigned int cpId = -1;

    const auto productID = simTS.seedID();
    if (productID == cPHandle_id) {
      cpId = simTS.seedIndex();
    } else {  // SimTrackster from SimCluster
      const auto findSimTSFromCP = std::find_if(
          std::begin(cpToSc_SimTrackstersMap),
          std::end(cpToSc_SimTrackstersMap),
          [&](const std::pair<unsigned int, std::vector<unsigned int>>& cpToScs) {
            return std::find(std::begin(cpToScs.second), std::end(cpToScs.second), iSTS) != std::end(cpToScs.second);
          });
      if (findSimTSFromCP != std::end(cpToSc_SimTrackstersMap)) {
        cpId = simTSs_fromCP[findSimTSFromCP->first].seedIndex();
      }
    }

    return cpId;
  };

  auto getLCId = [](const std::vector<unsigned int>& tst_vertices,
                    const reco::CaloClusterCollection& layerClusters,
                    const DetId& hitid) {
    unsigned int lcId = -1;
    std::for_each(std::begin(tst_vertices), std::end(tst_vertices), [&](unsigned int idx) {
      const auto& lc_haf = layerClusters[idx].hitsAndFractions();
      const auto& hitFound = std::find_if(std::begin(lc_haf),
                                          std::end(lc_haf),
                                          [&hitid](const std::pair<DetId, float>& v) { return v.first == hitid; });
      if (hitFound != lc_haf.end())  // not all hits may be clusterized
        lcId = idx;
    });
    return lcId;
  };

  for (unsigned int iSTS = 0; iSTS < nSimTracksters; ++iSTS) {
    const auto cpId = getCPId(simTSs[iSTS], iSTS, cPHandle_id, cpToSc_SimTrackstersMap, simTSs_fromCP);
    if (std::find(cPIndices.begin(), cPIndices.end(), cpId) == cPIndices.end())
      continue;

    // Loop through SimClusters
    for (const auto& simCluster : cP[cpId].simClusters()) {
      auto iSim = simTSs[iSTS].seedIndex();
      if (simTSs[iSTS].seedID() != cPHandle_id) {  // SimTrackster from SimCluster
        if (iSim != (&(*simCluster) - &(sC[0])))
          continue;
      } else
        iSim = 0;

      for (const auto& it_haf : simCluster->hits_and_fractions()) {
        const auto hitid = (it_haf.first);
        const auto lcId = getLCId(simTSs[iSTS].vertices(), layerClusters, hitid);
        //V9:maps the layers in -z: 0->51 and in +z: 52->103
        //V10:maps the layers in -z: 0->49 and in +z: 50->99
        const auto itcheck = hitMap.find(hitid);
        //Check whether the current hit belonging to sim cluster has a reconstructed hit
        if ((valType == 0 && itcheck != hitMap.end()) || (valType > 0 && int(lcId) >= 0)) {
          float lcFraction = 0;
          if (valType > 0) {
            const auto iLC = std::find(simTSs[iSTS].vertices().begin(), simTSs[iSTS].vertices().end(), lcId);
            lcFraction =
                1.f / simTSs[iSTS].vertex_multiplicity(std::distance(std::begin(simTSs[iSTS].vertices()), iLC));
          }
          const auto elemId = (valType == 0) ? hitid : lcId;
          const auto elemFr = (valType == 0) ? it_haf.second : lcFraction;
          //Since the current hit from SimCluster has a reconstructed hit {with the same detid, belonging to the corresponding SimTrackster},
          //make a map that will connect a {detid,lcID} with:
          //1. the SimTracksters that have a SimCluster with {sim hits in, LCs containing} that cell via SimTrackster id.
          //2. the sum of all {SimHits, LCs} fractions that contributes to that detid.
          //So, keep in mind that in case of multiple CaloParticles contributing in the same cell
          //the fraction is the sum over all calo particles. So, something like:
          //detid: (caloparticle 1, sum of hits fractions in that detid over all cp) , (caloparticle 2, sum of hits fractions in that detid over all cp), (caloparticle 3, sum of hits fractions in that detid over all cp) ...
          if (detIdSimTSId_Map.find(elemId) == detIdSimTSId_Map.end()) {
            detIdSimTSId_Map[elemId] = std::vector<HGVHistoProducerAlgo::detIdInfoInCluster>();
            detIdSimTSId_Map[elemId].emplace_back(HGVHistoProducerAlgo::detIdInfoInCluster{iSTS, elemFr});
          } else {
            auto findSTSIt =
                std::find(detIdSimTSId_Map[elemId].begin(),
                          detIdSimTSId_Map[elemId].end(),
                          HGVHistoProducerAlgo::detIdInfoInCluster{
                              iSTS, 0});  // only the first element is used for the matching (overloaded operator==)
            if (findSTSIt != detIdSimTSId_Map[elemId].end()) {
              if (valType == 0)
                findSTSIt->fraction += elemFr;
            } else {
              detIdSimTSId_Map[elemId].emplace_back(HGVHistoProducerAlgo::detIdInfoInCluster{iSTS, elemFr});
            }
          }
          const auto hitEn = itcheck->second->energy();
          //Since the current hit from sim cluster has a reconstructed hit with the same detid,
          //fill the cPOnLayer[caloparticle][layer] object with energy (sum of all rechits energy times fraction
          //of the relevant simhit) and keep the hit (detid and fraction) that contributed.
          cPOnLayer[cpId].energy += it_haf.second * hitEn;
          sCOnLayer[cpId][iSim].energy += elemFr * hitEn;
          // Need to compress the hits and fractions in order to have a
          // reasonable score between CP and LC. Imagine, for example, that a
          // CP has detID X used by 2 SimClusters with different fractions. If
          // a single LC uses X with fraction 1 and is compared to the 2
          // contributions separately, it will be assigned a score != 0, which
          // is wrong.
          auto& haf = cPOnLayer[cpId].hits_and_fractions;
          auto found = std::find_if(
              std::begin(haf), std::end(haf), [&hitid](const std::pair<DetId, float>& v) { return v.first == hitid; });
          if (found != haf.end())
            found->second += it_haf.second;
          else
            haf.emplace_back(hitid, it_haf.second);
          // Same for sCOnLayer
          auto& haf_sc = sCOnLayer[cpId][iSim].hits_and_fractions;
          auto found_sc = std::find_if(std::begin(haf_sc),
                                       std::end(haf_sc),
                                       [&hitid](const std::pair<DetId, float>& v) { return v.first == hitid; });
          if (found_sc != haf_sc.end())
            found_sc->second += it_haf.second;
          else
            haf_sc.emplace_back(hitid, it_haf.second);
        }
      }  // end of loop through SimHits
    }    // end of loop through SimClusters
  }      // end of loop through SimTracksters

  auto apply_LCMultiplicity = [](const ticl::Trackster& trackster, const reco::CaloClusterCollection& layerClusters) {
    std::vector<std::pair<DetId, float>> hits_and_fractions_norm;
    int lcInTst = 0;
    std::for_each(std::begin(trackster.vertices()), std::end(trackster.vertices()), [&](unsigned int idx) {
      const auto fraction = 1.f / trackster.vertex_multiplicity(lcInTst++);
      for (const auto& cell : layerClusters[idx].hitsAndFractions()) {
        hits_and_fractions_norm.emplace_back(
            cell.first, cell.second * fraction);  // cell.second is the hit fraction in the layerCluster
      }
    });
    return hits_and_fractions_norm;
  };

  auto ScoreCutSTStoTSPurDup = ScoreCutSTStoTSPurDup_[0];
  auto ScoreCutTStoSTSFakeMerge = ScoreCutTStoSTSFakeMerge_[0];
  // Loop through Tracksters
  for (unsigned int tstId = 0; tstId < nTracksters; ++tstId) {
    const auto& tst = tracksters[tstId];
    if (tstId == 0)
      if ((valType > 0) && (tst.ticlIteration() == ticl::Trackster::SIM)) {
        ScoreCutSTStoTSPurDup = ScoreCutSTStoTSPurDup_[valType];
        ScoreCutTStoSTSFakeMerge = ScoreCutTStoSTSFakeMerge_[valType];
      }

    if (tst.vertices().empty())
      continue;

    std::unordered_map<unsigned, float> CPEnergyInTS;
    int maxCPId_byNumberOfHits = -1;
    unsigned int maxCPNumberOfHitsInTS = 0;
    int maxCPId_byEnergy = -1;
    float maxEnergySharedTSandCP = 0.f;
    float energyFractionOfTSinCP = 0.f;
    float energyFractionOfCPinTS = 0.f;

    //In case of matched rechit-simhit, so matched
    //CaloParticle-LayerCluster-Trackster, he counts and saves the number of
    //rechits related to the maximum energy CaloParticle out of all
    //CaloParticles related to that layer cluster and Trackster.

    std::unordered_map<unsigned, unsigned> occurrencesCPinTS;
    unsigned int numberOfNoiseHitsInTS = 0;

    const auto tst_hitsAndFractions = apply_LCMultiplicity(tst, layerClusters);
    const auto numberOfHitsInTS = tst_hitsAndFractions.size();

    //hitsToCaloParticleId is a vector of ints, one for each rechit of the
    //layer cluster under study. If negative, there is no simhit from any CaloParticle related.
    //If positive, at least one CaloParticle has been found with matched simhit.
    //In more detail:
    // 1. hitsToCaloParticleId[iHit] = -3
    //    TN:  These represent Halo Cells(N) that have not been
    //    assigned to any CaloParticle (hence the T).
    // 2. hitsToCaloParticleId[iHit] = -2
    //    FN: There represent Halo Cells(N) that have been assigned
    //    to a CaloParticle (hence the F, since those should have not been marked as halo)
    // 3. hitsToCaloParticleId[iHit] = -1
    //    FP: These represent Real Cells(P) that have not been
    //    assigned to any CaloParticle (hence the F, since these are fakes)
    // 4. hitsToCaloParticleId[iHit] >= 0
    //    TP There represent Real Cells(P) that have been assigned
    //    to a CaloParticle (hence the T)
    std::vector<int> hitsToCaloParticleId(numberOfHitsInTS);

    //Loop through the hits of the trackster under study
    for (unsigned int iHit = 0; iHit < numberOfHitsInTS; iHit++) {
      const auto rh_detid = tst_hitsAndFractions[iHit].first;
      const auto rhFraction = tst_hitsAndFractions[iHit].second;

      const auto lcId_r = getLCId(tst.vertices(), layerClusters, rh_detid);
      const auto iLC_r = std::find(tst.vertices().begin(), tst.vertices().end(), lcId_r);
      const auto lcFraction_r = 1.f / tst.vertex_multiplicity(std::distance(std::begin(tst.vertices()), iLC_r));

      //Make a map that will connect a detid (that belongs to a rechit of the layer cluster under study,
      //no need to save others) with:
      //1. the layer clusters that have rechits in that detid
      //2. the fraction of the rechit of each layer cluster that contributes to that detid.
      //So, something like:
      //detid: (layer cluster 1, hit fraction) , (layer cluster 2, hit fraction), (layer cluster 3, hit fraction) ...
      //here comparing with the calo particle map above the
      if (detIdToTracksterId_Map.find(rh_detid) == detIdToTracksterId_Map.end()) {
        detIdToTracksterId_Map[rh_detid] = std::vector<HGVHistoProducerAlgo::detIdInfoInTrackster>();
        detIdToTracksterId_Map[rh_detid].emplace_back(
            HGVHistoProducerAlgo::detIdInfoInTrackster{tstId, lcId_r, rhFraction});
      } else {
        auto findTSIt =
            std::find(detIdToTracksterId_Map[rh_detid].begin(),
                      detIdToTracksterId_Map[rh_detid].end(),
                      HGVHistoProducerAlgo::detIdInfoInTrackster{
                          tstId, 0, 0});  // only the first element is used for the matching (overloaded operator==)
        if (findTSIt != detIdToTracksterId_Map[rh_detid].end()) {
          if (valType == 0)
            findTSIt->fraction += rhFraction;
        } else {
          detIdToTracksterId_Map[rh_detid].emplace_back(
              HGVHistoProducerAlgo::detIdInfoInTrackster{tstId, lcId_r, rhFraction});
        }
      }

      // if the fraction is zero or the hit does not belong to any calo
      // particle, set the caloparticleId for the hit to -1 this will
      // contribute to the number of noise hits
      // MR Remove the case in which the fraction is 0, since this could be a
      // real hit that has been marked as halo.
      if (rhFraction == 0.) {
        hitsToCaloParticleId[iHit] = -2;
      }

      // Check whether the RecHit of the trackster under study has a SimHit in the same cell
      const auto elemId = (valType == 0) ? rh_detid.rawId() : lcId_r;
      const auto recoFr = (valType == 0) ? rhFraction : lcFraction_r;
      const auto& hit_find_in_STS = detIdSimTSId_Map.find(elemId);
      if (hit_find_in_STS == detIdSimTSId_Map.end()) {
        hitsToCaloParticleId[iHit] -= 1;
      } else {
        // Since the hit is belonging to the layer cluster, it must be also in the rechits map
        const auto hitEn = hitMap.find(rh_detid)->second->energy();
        //const auto layerId =
        //recHitTools_->getLayerWithOffset(rh_detid) + layers * ((recHitTools_->zside(rh_detid) + 1) >> 1) - 1;
        //0;

        auto maxCPEnergyInTS = 0.f;
        auto maxCPId = -1;
        for (const auto& h : hit_find_in_STS->second) {
          const auto shared_fraction = std::min(recoFr, h.fraction);
          const auto iSTS = h.clusterId;
          const auto& simTS = simTSs[iSTS];
          auto iSim = simTS.seedIndex();
          if (simTSs[iSTS].seedID() == cPHandle_id)  // SimTrackster from CaloParticle
            iSim = 0;

          // SimTrackster with simHits connected via detid with the rechit under study
          //So, from all layers clusters, find the rechits that are connected with a calo particle and save/calculate the
          //energy of that calo particle as the sum over all rechits of the rechits energy weighted
          //by the caloparticle's fraction related to that rechit.
          const auto cpId = getCPId(simTS, iSTS, cPHandle_id, cpToSc_SimTrackstersMap, simTSs_fromCP);
          if (std::find(cPIndices.begin(), cPIndices.end(), cpId) == cPIndices.end())
            continue;

          CPEnergyInTS[cpId] += shared_fraction * hitEn;
          //Here cPOnLayer[caloparticle][layer] describe above is set.
          //Here for Tracksters with matched rechit the CP fraction times hit energy is added and saved .
          cPOnLayer[cpId].layerClusterIdToEnergyAndScore[tstId].first += shared_fraction * hitEn;
          sCOnLayer[cpId][iSim].layerClusterIdToEnergyAndScore[tstId].first += shared_fraction * hitEn;
          cPOnLayer[cpId].layerClusterIdToEnergyAndScore[tstId].second = FLT_MAX;
          sCOnLayer[cpId][iSim].layerClusterIdToEnergyAndScore[tstId].second = FLT_MAX;
          //stsInTrackster[trackster][STSids]
          //Connects a Trackster with all related SimTracksters.
          stsInTrackster[tstId].emplace_back(iSTS, FLT_MAX);
          //From all CaloParticles related to a layer cluster, it saves id and energy of the calo particle
          //that after simhit-rechit matching in layer has the maximum energy.
          if (shared_fraction > maxCPEnergyInTS) {
            //energy is used only here. cpid is saved for Tracksters
            maxCPEnergyInTS = CPEnergyInTS[cpId];
            maxCPId = cpId;
          }
        }
        //Keep in mind here maxCPId could be zero. So, below ask for negative not including zero to count noise.
        hitsToCaloParticleId[iHit] = maxCPId;
      }

    }  //end of loop through rechits of the layer cluster.

    //Loop through all rechits to count how many of them are noise and how many are matched.
    //In case of matched rechit-simhit, he counts and saves the number of rechits related to the maximum energy CaloParticle.
    for (auto c : hitsToCaloParticleId) {
      if (c < 0)
        numberOfNoiseHitsInTS++;
      else
        occurrencesCPinTS[c]++;
    }

    //Below from all maximum energy CaloParticles, he saves the one with the largest amount
    //of related rechits.
    for (auto& c : occurrencesCPinTS) {
      if (c.second > maxCPNumberOfHitsInTS) {
        maxCPId_byNumberOfHits = c.first;
        maxCPNumberOfHitsInTS = c.second;
      }
    }

    //Find the CaloParticle that has the maximum energy shared with the Trackster under study.
    for (auto& c : CPEnergyInTS) {
      if (c.second > maxEnergySharedTSandCP) {
        maxCPId_byEnergy = c.first;
        maxEnergySharedTSandCP = c.second;
      }
    }
    //The energy of the CaloParticle that found to have the maximum energy shared with the Trackster under study.
    float totalCPEnergyFromLayerCP = 0.f;
    if (maxCPId_byEnergy >= 0) {
      totalCPEnergyFromLayerCP += cPOnLayer[maxCPId_byEnergy].energy;
      energyFractionOfCPinTS = maxEnergySharedTSandCP / totalCPEnergyFromLayerCP;
      if (tst.raw_energy() > 0.f) {
        energyFractionOfTSinCP = maxEnergySharedTSandCP / tst.raw_energy();
      }
    }

    LogDebug("HGCalValidator") << std::setw(12) << "Trackster\t" << std::setw(10) << "energy\t" << std::setw(5)
                               << "nhits\t" << std::setw(12) << "noise hits\t" << std::setw(22)
                               << "maxCPId_byNumberOfHits\t" << std::setw(8) << "nhitsCP\t" << std::setw(16)
                               << "maxCPId_byEnergy\t" << std::setw(23) << "maxEnergySharedTSandCP\t" << std::setw(22)
                               << "totalCPEnergyFromAllLayerCP\t" << std::setw(22) << "energyFractionOfTSinCP\t"
                               << std::setw(25) << "energyFractionOfCPinTS\t" << std::endl;
    LogDebug("HGCalValidator") << std::setw(12) << tstId << "\t"  //LogDebug("HGCalValidator")
                               << std::setw(10) << tst.raw_energy() << "\t" << std::setw(5) << numberOfHitsInTS << "\t"
                               << std::setw(12) << numberOfNoiseHitsInTS << "\t" << std::setw(22)
                               << maxCPId_byNumberOfHits << "\t" << std::setw(8) << maxCPNumberOfHitsInTS << "\t"
                               << std::setw(16) << maxCPId_byEnergy << "\t" << std::setw(23) << maxEnergySharedTSandCP
                               << "\t" << std::setw(22) << totalCPEnergyFromLayerCP << "\t" << std::setw(22)
                               << energyFractionOfTSinCP << "\t" << std::setw(25) << energyFractionOfCPinTS
                               << std::endl;

  }  // end of loop through Tracksters

  // Loop through Tracksters
  for (unsigned int tstId = 0; tstId < nTracksters; ++tstId) {
    const auto& tst = tracksters[tstId];
    if (tst.vertices().empty())
      continue;

    // find the unique SimTrackster ids contributing to the Trackster
    //stsInTrackster[trackster][STSids]
    std::sort(stsInTrackster[tstId].begin(), stsInTrackster[tstId].end());
    const auto last = std::unique(stsInTrackster[tstId].begin(), stsInTrackster[tstId].end());
    stsInTrackster[tstId].erase(last, stsInTrackster[tstId].end());

    if (tst.raw_energy() == 0. && !stsInTrackster[tstId].empty()) {
      //Loop through all SimTracksters contributing to Trackster tstId
      for (auto& stsPair : stsInTrackster[tstId]) {
        // In case of a Trackster with zero energy but related SimTracksters the score is set to 1
        stsPair.second = 1.;
        LogDebug("HGCalValidator") << "Trackster Id:\t" << tstId << "\tSimTrackster id:\t" << stsPair.first
                                   << "\tscore\t" << stsPair.second << std::endl;
        histograms.h_score_trackster2caloparticle[valType][count]->Fill(stsPair.second);
      }
      continue;
    }

    const auto tst_hitsAndFractions = apply_LCMultiplicity(tst, layerClusters);

    // Compute the correct normalization
    float tracksterEnergy = 0.f, invTracksterEnergyWeight = 0.f;
    for (const auto& haf : tst_hitsAndFractions) {
      float hitFr = 0.f;
      if (valType == 0) {
        hitFr = haf.second;
      } else {
        const auto lcId = getLCId(tst.vertices(), layerClusters, haf.first);
        const auto iLC = std::find(tst.vertices().begin(), tst.vertices().end(), lcId);
        hitFr = 1.f / tst.vertex_multiplicity(std::distance(std::begin(tst.vertices()), iLC));
      }
      tracksterEnergy += hitFr * hitMap.at(haf.first)->energy();
      invTracksterEnergyWeight += pow(hitFr * hitMap.at(haf.first)->energy(), 2);
    }
    if (invTracksterEnergyWeight)
      invTracksterEnergyWeight = 1.f / invTracksterEnergyWeight;

    for (const auto& haf : tst_hitsAndFractions) {
      const auto rh_detid = haf.first;
      unsigned int elemId = 0;
      float rhFraction = 0.f;
      if (valType == 0) {
        elemId = rh_detid.rawId();
        rhFraction = haf.second;
      } else {
        const auto lcId = getLCId(tst.vertices(), layerClusters, rh_detid);
        elemId = lcId;
        const auto iLC = std::find(tst.vertices().begin(), tst.vertices().end(), lcId);
        rhFraction = 1.f / tst.vertex_multiplicity(std::distance(std::begin(tst.vertices()), iLC));
      }

      bool hitWithNoSTS = false;
      if (detIdSimTSId_Map.find(elemId) == detIdSimTSId_Map.end())
        hitWithNoSTS = true;
      const HGCRecHit* hit = hitMap.find(rh_detid)->second;
      const auto hitEnergyWeight = pow(hit->energy(), 2);

      for (auto& stsPair : stsInTrackster[tstId]) {
        float cpFraction = 0.f;
        if (!hitWithNoSTS) {
          const auto& findSTSIt = std::find(
              detIdSimTSId_Map[elemId].begin(),
              detIdSimTSId_Map[elemId].end(),
              HGVHistoProducerAlgo::detIdInfoInCluster{
                  stsPair.first, 0.f});  // only the first element is used for the matching (overloaded operator==)
          if (findSTSIt != detIdSimTSId_Map[elemId].end())
            cpFraction = findSTSIt->fraction;
        }
        if (stsPair.second == FLT_MAX) {
          stsPair.second = 0.f;
        }
        stsPair.second +=
            min(pow(rhFraction - cpFraction, 2), pow(rhFraction, 2)) * hitEnergyWeight * invTracksterEnergyWeight;
      }
    }  // end of loop through trackster rechits

    //In case of a Trackster with some energy but none related CaloParticles print some info.
    if (stsInTrackster[tstId].empty())
      LogDebug("HGCalValidator") << "Trackster Id: " << tstId << "\tSimTrackster id: -1"
                                 << "\tscore: -1\n";

    tracksters_FakeMerge[tstId] =
        std::count_if(std::begin(stsInTrackster[tstId]),
                      std::end(stsInTrackster[tstId]),
                      [ScoreCutTStoSTSFakeMerge](const auto& obj) { return obj.second < ScoreCutTStoSTSFakeMerge; });

    const auto score = std::min_element(std::begin(stsInTrackster[tstId]),
                                        std::end(stsInTrackster[tstId]),
                                        [](const auto& obj1, const auto& obj2) { return obj1.second < obj2.second; });
    float score2 = -1;
    float sharedEneFrac2 = 0;
    for (const auto& stsPair : stsInTrackster[tstId]) {
      const auto iSTS = stsPair.first;
      const auto iScore = stsPair.second;
      const auto cpId = getCPId(simTSs[iSTS], iSTS, cPHandle_id, cpToSc_SimTrackstersMap, simTSs_fromCP);
      auto iSim = simTSs[iSTS].seedIndex();
      if (simTSs[iSTS].seedID() == cPHandle_id)  // SimTrackster from CaloParticle
        iSim = 0;
      const auto& simOnLayer = (valType == 0) ? cPOnLayer[cpId] : sCOnLayer[cpId][iSim];

      float sharedeneCPallLayers = 0.;
      sharedeneCPallLayers += simOnLayer.layerClusterIdToEnergyAndScore.count(tstId)
                                  ? simOnLayer.layerClusterIdToEnergyAndScore.at(tstId).first
                                  : 0;
      if (tracksterEnergy == 0)
        continue;
      const auto sharedEneFrac = sharedeneCPallLayers / tracksterEnergy;
      LogDebug("HGCalValidator") << "\nTrackster id: " << tstId << " (" << tst.vertices().size() << " vertices)"
                                 << "\tSimTrackster Id: " << iSTS << " (" << simTSs[iSTS].vertices().size()
                                 << " vertices)"
                                 << " (CP id: " << cpId << ")\tscore: " << iScore
                                 << "\tsharedeneCPallLayers: " << sharedeneCPallLayers << std::endl;

      histograms.h_score_trackster2caloparticle[valType][count]->Fill(iScore);
      histograms.h_sharedenergy_trackster2caloparticle[valType][count]->Fill(sharedEneFrac);
      histograms.h_energy_vs_score_trackster2caloparticle[valType][count]->Fill(iScore, sharedEneFrac);
      if (iSTS == score->first) {
        histograms.h_score_trackster2bestCaloparticle[valType][count]->Fill(iScore);
        histograms.h_sharedenergy_trackster2bestCaloparticle[valType][count]->Fill(sharedEneFrac);
        histograms.h_sharedenergy_trackster2bestCaloparticle_vs_eta[valType][count]->Fill(tst.barycenter().eta(),
                                                                                          sharedEneFrac);
        histograms.h_sharedenergy_trackster2bestCaloparticle_vs_phi[valType][count]->Fill(tst.barycenter().phi(),
                                                                                          sharedEneFrac);
        histograms.h_energy_vs_score_trackster2bestCaloparticle[valType][count]->Fill(iScore, sharedEneFrac);
      } else if (score2 < 0 || iScore < score2) {
        score2 = iScore;
        sharedEneFrac2 = sharedEneFrac;
      }
    }  // end of loop through SimTracksters associated to Trackster
    if (score2 > -1) {
      histograms.h_score_trackster2bestCaloparticle2[valType][count]->Fill(score2);
      histograms.h_sharedenergy_trackster2bestCaloparticle2[valType][count]->Fill(sharedEneFrac2);
      histograms.h_energy_vs_score_trackster2bestCaloparticle2[valType][count]->Fill(score2, sharedEneFrac2);
    }
  }  // end of loop through Tracksters

  std::unordered_map<unsigned int, std::vector<float>> score3d;
  std::unordered_map<unsigned int, std::vector<float>> tstSharedEnergy;

  for (unsigned int iSTS = 0; iSTS < nSimTracksters; ++iSTS) {
    score3d[iSTS].resize(nTracksters);
    tstSharedEnergy[iSTS].resize(nTracksters);
    for (unsigned int j = 0; j < nTracksters; ++j) {
      score3d[iSTS][j] = FLT_MAX;
      tstSharedEnergy[iSTS][j] = 0.f;
    }
  }

  // Fill the plots to compute the different metrics linked to
  // gen-level, namely efficiency, purity and duplicate. In this loop should restrict
  // only to the selected caloParaticles.
  for (unsigned int iSTS = 0; iSTS < nSimTracksters; ++iSTS) {
    const auto& sts = simTSs[iSTS];
    const auto& cpId = getCPId(sts, iSTS, cPHandle_id, cpToSc_SimTrackstersMap, simTSs_fromCP);
    if (valType == 0 && std::find(cPSelectedIndices.begin(), cPSelectedIndices.end(), cpId) == cPSelectedIndices.end())
      continue;

    const auto& hafLC = apply_LCMultiplicity(sts, layerClusters);
    float SimEnergy_LC = 0.f;
    for (const auto& haf : hafLC) {
      const auto lcId = getLCId(sts.vertices(), layerClusters, haf.first);
      const auto iLC = std::find(sts.vertices().begin(), sts.vertices().end(), lcId);
      SimEnergy_LC +=
          hitMap.at(haf.first)->energy() / sts.vertex_multiplicity(std::distance(std::begin(sts.vertices()), iLC));
    }

    auto iSim = sts.seedIndex();
    if (sts.seedID() == cPHandle_id)  // SimTrackster from CaloParticle
      iSim = 0;
    auto& simOnLayer = (valType == 0) ? cPOnLayer[cpId] : sCOnLayer[cpId][iSim];

    // Keep the Trackster ids that are related to
    // SimTrackster under study for the final filling of the score
    std::set<unsigned int> stsId_tstId_related;
    auto& score3d_iSTS = score3d[iSTS];

    float SimEnergy = 0.f;
    float SimEnergyWeight = 0.f, hitsEnergyWeight = 0.f;
    //for (unsigned int layerId = 0; layerId < 1/*layers * 2*/; ++layerId) {
    const auto SimNumberOfHits = simOnLayer.hits_and_fractions.size();
    if (SimNumberOfHits == 0)
      continue;
    SimEnergy += simOnLayer.energy;
    int tstWithMaxEnergyInCP = -1;
    //This is the maximum energy related to Trackster per layer.
    float maxEnergyTSperlayerinSim = 0.f;
    float SimEnergyFractionInTSperlayer = 0.f;
    //Remember and not confused by name. layerClusterIdToEnergyAndScore contains the Trackster id.
    for (const auto& tst : simOnLayer.layerClusterIdToEnergyAndScore) {
      if (tst.second.first > maxEnergyTSperlayerinSim) {
        maxEnergyTSperlayerinSim = tst.second.first;
        tstWithMaxEnergyInCP = tst.first;
      }
    }
    if (SimEnergy > 0.f)
      SimEnergyFractionInTSperlayer = maxEnergyTSperlayerinSim / SimEnergy;

    LogDebug("HGCalValidator") << std::setw(12) << "caloparticle\t" << std::setw(15) << "cp total energy\t"
                               << std::setw(15) << "cpEnergyOnLayer\t" << std::setw(14) << "CPNhitsOnLayer\t"
                               << std::setw(18) << "tstWithMaxEnergyInCP\t" << std::setw(15) << "maxEnergyTSinCP\t"
                               << std::setw(20) << "CPEnergyFractionInTS"
                               << "\n";
    LogDebug("HGCalValidator") << std::setw(12) << cpId << "\t" << std::setw(15) << sts.raw_energy() << "\t"
                               << std::setw(15) << SimEnergy << "\t" << std::setw(14) << SimNumberOfHits << "\t"
                               << std::setw(18) << tstWithMaxEnergyInCP << "\t" << std::setw(15)
                               << maxEnergyTSperlayerinSim << "\t" << std::setw(20) << SimEnergyFractionInTSperlayer
                               << "\n";

    for (const auto& haf : ((valType == 0) ? simOnLayer.hits_and_fractions : hafLC)) {
      const auto& hitDetId = haf.first;
      // Compute the correct normalization
      // Need to loop on the simOnLayer data structure since this is the
      // only one that has the compressed information for multiple usage
      // of the same DetId by different SimClusters by a single CaloParticle.
      SimEnergyWeight += pow(haf.second * hitMap.at(hitDetId)->energy(), 2);

      const auto lcId = getLCId(sts.vertices(), layerClusters, hitDetId);
      float cpFraction = 0.f;
      if (valType == 0) {
        cpFraction = haf.second;
      } else {
        const auto iLC = std::find(sts.vertices().begin(), sts.vertices().end(), lcId);
        cpFraction = 1.f / sts.vertex_multiplicity(std::distance(std::begin(sts.vertices()), iLC));
      }
      if (cpFraction == 0.f)
        continue;  // hopefully this should never happen

      bool hitWithNoTS = false;
      if (detIdToTracksterId_Map.find(hitDetId) == detIdToTracksterId_Map.end())
        hitWithNoTS = true;
      const HGCRecHit* hit = hitMap.find(hitDetId)->second;
      const auto hitEnergyWeight = pow(hit->energy(), 2);
      hitsEnergyWeight += pow(cpFraction, 2) * hitEnergyWeight;

      for (auto& tsPair : simOnLayer.layerClusterIdToEnergyAndScore) {
        const auto tstId = tsPair.first;
        stsId_tstId_related.insert(tstId);

        float tstFraction = 0.f;
        if (!hitWithNoTS) {
          const auto findTSIt =
              std::find(detIdToTracksterId_Map[hitDetId].begin(),
                        detIdToTracksterId_Map[hitDetId].end(),
                        HGVHistoProducerAlgo::detIdInfoInTrackster{
                            tstId, 0, 0.f});  // only the first element is used for the matching (overloaded operator==)
          if (findTSIt != detIdToTracksterId_Map[hitDetId].end()) {
            if (valType == 0) {
              tstFraction = findTSIt->fraction;
            } else {
              const auto iLC = std::find(
                  tracksters[tstId].vertices().begin(), tracksters[tstId].vertices().end(), findTSIt->clusterId);
              if (iLC != tracksters[tstId].vertices().end()) {
                tstFraction = 1.f / tracksters[tstId].vertex_multiplicity(
                                        std::distance(std::begin(tracksters[tstId].vertices()), iLC));
              }
            }
          }
        }
        // Here do not divide as before by the trackster energy weight. Should sum first
        // over all layers and divide with the total CP energy over all layers.
        if (tsPair.second.second == FLT_MAX) {
          tsPair.second.second = 0.f;
        }
        tsPair.second.second += min(pow(tstFraction - cpFraction, 2), pow(cpFraction, 2)) * hitEnergyWeight;

        LogDebug("HGCalValidator") << "\nTracksterId:\t" << tstId << "\tSimTracksterId:\t" << iSTS << "\tcpId:\t"
                                   << cpId << "\ttstfraction, cpfraction:\t" << tstFraction << ", " << cpFraction
                                   << "\thitEnergyWeight:\t" << hitEnergyWeight << "\tadded delta:\t"
                                   << pow((tstFraction - cpFraction), 2) * hitEnergyWeight
                                   << "\tcurrent Sim-score numerator:\t" << tsPair.second.second
                                   << "\tshared Sim energy:\t" << tsPair.second.first << '\n';
      }
    }  // end of loop through SimCluster SimHits on current layer

    if (simOnLayer.layerClusterIdToEnergyAndScore.empty())
      LogDebug("HGCalValidator") << "CP Id:\t" << cpId << "\tTS id:\t-1"
                                 << " Sub score in \t -1\n";

    for (const auto& tsPair : simOnLayer.layerClusterIdToEnergyAndScore) {
      const auto tstId = tsPair.first;
      // 3D score here without the denominator at this point
      if (score3d_iSTS[tstId] == FLT_MAX) {
        score3d_iSTS[tstId] = 0.f;
      }
      score3d_iSTS[tstId] += tsPair.second.second;
      tstSharedEnergy[iSTS][tstId] += tsPair.second.first;
    }
    //} // end of loop through layers

    const auto scoreDenom = (valType == 0) ? SimEnergyWeight : hitsEnergyWeight;
    const auto energyDenom = (valType == 0) ? SimEnergy : SimEnergy_LC;

    const auto sts_eta = sts.barycenter().eta();
    const auto sts_phi = sts.barycenter().phi();
    const auto sts_en = sts.raw_energy();
    const auto sts_pt = sts.raw_pt();
    histograms.h_denom_caloparticle_eta[valType][count]->Fill(sts_eta);
    histograms.h_denom_caloparticle_phi[valType][count]->Fill(sts_phi);
    histograms.h_denom_caloparticle_en[valType][count]->Fill(sts_en);
    histograms.h_denom_caloparticle_pt[valType][count]->Fill(sts_pt);

    //Loop through related Tracksters here
    // In case the threshold to associate a CaloParticle to a Trackster is
    // below 50%, there could be cases in which the CP is linked to more than
    // one tracksters, leading to efficiencies >1. This boolean is used to
    // avoid "over counting".
    bool sts_considered_efficient = false;
    bool sts_considered_pure = false;
    for (const auto tstId : stsId_tstId_related) {
      // Now time for the denominator
      score3d_iSTS[tstId] /= scoreDenom;
      const auto tstSharedEnergyFrac = tstSharedEnergy[iSTS][tstId] / energyDenom;
      LogDebug("HGCalValidator") << "STS id: " << iSTS << "\t(CP id: " << cpId << ")\tTS id: " << tstId
                                 << "\nSimEnergy: " << energyDenom << "\tSimEnergyWeight: " << SimEnergyWeight
                                 << "\tTrackste energy: " << tracksters[tstId].raw_energy()
                                 << "\nscore: " << score3d_iSTS[tstId]
                                 << "\tshared energy: " << tstSharedEnergy[iSTS][tstId]
                                 << "\tshared energy fraction: " << tstSharedEnergyFrac << "\n";

      histograms.h_score_caloparticle2trackster[valType][count]->Fill(score3d_iSTS[tstId]);
      histograms.h_sharedenergy_caloparticle2trackster[valType][count]->Fill(tstSharedEnergyFrac);
      histograms.h_energy_vs_score_caloparticle2trackster[valType][count]->Fill(score3d_iSTS[tstId],
                                                                                tstSharedEnergyFrac);
      // Fill the numerator for the efficiency calculation. The efficiency is computed by considering the energy shared between a Trackster and a _corresponding_ caloParticle. The threshold is configurable via python.
      if (!sts_considered_efficient && (tstSharedEnergyFrac >= minTSTSharedEneFracEfficiency_)) {
        sts_considered_efficient = true;
        histograms.h_numEff_caloparticle_eta[valType][count]->Fill(sts_eta);
        histograms.h_numEff_caloparticle_phi[valType][count]->Fill(sts_phi);
        histograms.h_numEff_caloparticle_en[valType][count]->Fill(sts_en);
        histograms.h_numEff_caloparticle_pt[valType][count]->Fill(sts_pt);
      }

      if (score3d_iSTS[tstId] < ScoreCutSTStoTSPurDup) {
        if (tracksters_PurityDuplicate[tstId] < 1)
          tracksters_PurityDuplicate[tstId]++;  // for Purity
        if (sts_considered_pure)
          tracksters_PurityDuplicate[tstId]++;  // for Duplicate
        sts_considered_pure = true;
      }
    }  // end of loop through Tracksters related to SimTrackster

    const auto best = std::min_element(std::begin(score3d_iSTS), std::end(score3d_iSTS));
    if (best != score3d_iSTS.end()) {
      const auto bestTstId = std::distance(std::begin(score3d_iSTS), best);
      const auto bestTstSharedEnergyFrac = tstSharedEnergy[iSTS][bestTstId] / energyDenom;
      histograms.h_scorePur_caloparticle2trackster[valType][count]->Fill(*best);
      histograms.h_sharedenergy_caloparticle2trackster_assoc[valType][count]->Fill(bestTstSharedEnergyFrac);
      histograms.h_sharedenergy_caloparticle2trackster_assoc_vs_eta[valType][count]->Fill(sts_eta,
                                                                                          bestTstSharedEnergyFrac);
      histograms.h_sharedenergy_caloparticle2trackster_assoc_vs_phi[valType][count]->Fill(sts_phi,
                                                                                          bestTstSharedEnergyFrac);
      histograms.h_energy_vs_score_caloparticle2bestTrackster[valType][count]->Fill(*best, bestTstSharedEnergyFrac);
      LogDebug("HGCalValidator") << count << " " << sts_eta << " " << sts_phi << " "
                                 << tracksters[bestTstId].raw_energy() << " " << sts.raw_energy() << " "
                                 << bestTstSharedEnergyFrac << "\n";

      if (score3d_iSTS.size() > 1) {
        auto best2 = (best == score3d_iSTS.begin()) ? std::next(best, 1) : score3d_iSTS.begin();
        for (auto tstId = score3d_iSTS.begin(); tstId != score3d_iSTS.end() && tstId != best; tstId++)
          if (*tstId < *best2)
            best2 = tstId;
        const auto best2TstId = std::distance(std::begin(score3d_iSTS), best2);
        const auto best2TstSharedEnergyFrac = tstSharedEnergy[iSTS][best2TstId] / energyDenom;
        histograms.h_scoreDupl_caloparticle2trackster[valType][count]->Fill(*best2);
        histograms.h_sharedenergy_caloparticle2trackster_assoc2[valType][count]->Fill(best2TstSharedEnergyFrac);
        histograms.h_energy_vs_score_caloparticle2bestTrackster2[valType][count]->Fill(*best2,
                                                                                       best2TstSharedEnergyFrac);
      }
    }
  }  // end of loop through SimTracksters

  // Fill the plots to compute the different metrics linked to
  // reco-level, namely fake-rate an merge-rate. Should *not*
  // restrict only to the selected caloParaticles.
  for (unsigned int tstId = 0; tstId < nTracksters; ++tstId) {
    const auto& tst = tracksters[tstId];
    if (tst.vertices().empty())
      continue;
    const auto iTS_eta = tst.barycenter().eta();
    const auto iTS_phi = tst.barycenter().phi();
    const auto iTS_en = tst.raw_energy();
    const auto iTS_pt = tst.raw_pt();
    histograms.h_denom_trackster_eta[valType][count]->Fill(iTS_eta);
    histograms.h_denom_trackster_phi[valType][count]->Fill(iTS_phi);
    histograms.h_denom_trackster_en[valType][count]->Fill(iTS_en);
    histograms.h_denom_trackster_pt[valType][count]->Fill(iTS_pt);

    if (tracksters_PurityDuplicate[tstId] > 0) {
      histograms.h_num_caloparticle_eta[valType][count]->Fill(iTS_eta);
      histograms.h_num_caloparticle_phi[valType][count]->Fill(iTS_phi);
      histograms.h_num_caloparticle_en[valType][count]->Fill(iTS_en);
      histograms.h_num_caloparticle_pt[valType][count]->Fill(iTS_pt);

      if (tracksters_PurityDuplicate[tstId] > 1) {
        histograms.h_numDup_trackster_eta[valType][count]->Fill(iTS_eta);
        histograms.h_numDup_trackster_phi[valType][count]->Fill(iTS_phi);
        histograms.h_numDup_trackster_en[valType][count]->Fill(iTS_en);
        histograms.h_numDup_trackster_pt[valType][count]->Fill(iTS_pt);
      }
    }

    if (tracksters_FakeMerge[tstId] > 0) {
      histograms.h_num_trackster_eta[valType][count]->Fill(iTS_eta);
      histograms.h_num_trackster_phi[valType][count]->Fill(iTS_phi);
      histograms.h_num_trackster_en[valType][count]->Fill(iTS_en);
      histograms.h_num_trackster_pt[valType][count]->Fill(iTS_pt);

      if (tracksters_FakeMerge[tstId] > 1) {
        histograms.h_numMerge_trackster_eta[valType][count]->Fill(iTS_eta);
        histograms.h_numMerge_trackster_phi[valType][count]->Fill(iTS_phi);
        histograms.h_numMerge_trackster_en[valType][count]->Fill(iTS_en);
        histograms.h_numMerge_trackster_pt[valType][count]->Fill(iTS_pt);
      }
    }
  }  // End loop over Tracksters
}

void HGVHistoProducerAlgo::fill_trackster_histos(
    const Histograms& histograms,
    const int count,
    const ticl::TracksterCollection& tracksters,
    const reco::CaloClusterCollection& layerClusters,
    const ticl::TracksterCollection& simTSs,
    const ticl::TracksterCollection& simTSs_fromCP,
    const std::map<unsigned int, std::vector<unsigned int>>& cpToSc_SimTrackstersMap,
    std::vector<SimCluster> const& sC,
    const edm::ProductID& cPHandle_id,
    std::vector<CaloParticle> const& cP,
    std::vector<size_t> const& cPIndices,
    std::vector<size_t> const& cPSelectedIndices,
    std::unordered_map<DetId, const HGCRecHit*> const& hitMap,
    unsigned int layers) const {
  //Each event to be treated as two events:
  //an event in +ve endcap, plus another event in -ve endcap.

  //To keep track of total num of Tracksters
  int totNTstZm = 0;  //-z
  int totNTstZp = 0;  //+z
  //To count the number of Tracksters with 3 contiguous layers per event.
  int totNContTstZp = 0;  //+z
  int totNContTstZm = 0;  //-z
  //For the number of Tracksters without 3 contiguous layers per event.
  int totNNotContTstZp = 0;  //+z
  int totNNotContTstZm = 0;  //-z
  // Check below the score of cont and non cont Tracksters
  std::vector<bool> contTracksters;
  contTracksters.clear();

  //[tstId]-> vector of 2d layer clusters size
  std::unordered_map<unsigned int, std::vector<unsigned int>> multiplicity;
  //[tstId]-> [layer][cluster size]
  std::unordered_map<unsigned int, std::vector<unsigned int>> multiplicity_vs_layer;
  //We will need for the scale text option
  // unsigned int totalLcInTsts = 0;
  // for (unsigned int tstId = 0; tstId < nTracksters; ++tstId) {
  //   totalLcInTsts = totalLcInTsts + tracksters[tstId].vertices().size();
  // }

  const auto nTracksters = tracksters.size();
  // loop through Tracksters
  for (unsigned int tstId = 0; tstId < nTracksters; ++tstId) {
    const auto& tst = tracksters[tstId];
    if (tst.vertices().empty())
      continue;

    if (tst.barycenter().z() < 0.)
      totNTstZm++;
    else if (tst.barycenter().z() > 0.)
      totNTstZp++;

    //Total number of layer clusters in Trackster
    int tnLcInTst = 0;

    //To keep track of total num of layer clusters per Trackster
    //tnLcInTstperlaypz[layerid], tnLcInTstperlaymz[layerid]
    std::vector<int> tnLcInTstperlay(1000, 0);  //+z

    //For the layers the Trackster expands to. Will use a set because there would be many
    //duplicates and then go back to vector for random access, since they say it is faster.
    std::set<unsigned int> trackster_layers;

    bool tracksterInZplus = false;
    bool tracksterInZminus = false;

    //Loop through layer clusters
    for (const auto lcId : tst.vertices()) {
      //take the hits and their fraction of the specific layer cluster.
      const auto& hits_and_fractions = layerClusters[lcId].hitsAndFractions();

      //For the multiplicity of the 2d layer clusters in Tracksters
      multiplicity[tstId].emplace_back(hits_and_fractions.size());

      const auto firstHitDetId = hits_and_fractions[0].first;
      //The layer that the layer cluster belongs to
      const auto layerid = recHitTools_->getLayerWithOffset(firstHitDetId) +
                           layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
      trackster_layers.insert(layerid);
      multiplicity_vs_layer[tstId].emplace_back(layerid);

      tnLcInTstperlay[layerid]++;
      tnLcInTst++;

      if (recHitTools_->zside(firstHitDetId) > 0.)
        tracksterInZplus = true;
      else if (recHitTools_->zside(firstHitDetId) < 0.)
        tracksterInZminus = true;
    }  // end of loop through layerClusters

    // Per layer : Loop 0->99
    for (unsigned ilayer = 0; ilayer < layers * 2; ++ilayer) {
      if (histograms.h_clusternum_in_trackster_perlayer[count].count(ilayer) && tnLcInTstperlay[ilayer] != 0) {
        histograms.h_clusternum_in_trackster_perlayer[count].at(ilayer)->Fill((float)tnLcInTstperlay[ilayer]);
      }
      // For the profile now of 2d layer cluster in Tracksters vs layer number.
      if (tnLcInTstperlay[ilayer] != 0) {
        histograms.h_clusternum_in_trackster_vs_layer[count]->Fill((float)ilayer, (float)tnLcInTstperlay[ilayer]);
      }
    }  // end of loop over layers

    // Looking for Tracksters with 3 contiguous layers per event.
    std::vector<int> trackster_layers_vec(trackster_layers.begin(), trackster_layers.end());
    // Check also for non contiguous Tracksters
    bool contiTrackster = false;
    // Start from 1 and go up to size - 1 element.
    if (trackster_layers_vec.size() >= 3) {
      for (unsigned int iLayer = 1; iLayer < trackster_layers_vec.size() - 1; ++iLayer) {
        if ((trackster_layers_vec[iLayer - 1] + 1 == trackster_layers_vec[iLayer]) &&
            (trackster_layers_vec[iLayer + 1] - 1 == trackster_layers_vec[iLayer])) {
          // Trackster with 3 contiguous layers per event
          if (tracksterInZplus)
            totNContTstZp++;
          else if (tracksterInZminus)
            totNContTstZm++;

          contiTrackster = true;
          break;
        }
      }
    }
    // Count non contiguous Tracksters
    if (!contiTrackster) {
      if (tracksterInZplus)
        totNNotContTstZp++;
      else if (tracksterInZminus)
        totNNotContTstZm++;
    }

    // Save for the score
    contTracksters.push_back(contiTrackster);

    histograms.h_clusternum_in_trackster[count]->Fill(tnLcInTst);

    for (unsigned int lc = 0; lc < multiplicity[tstId].size(); ++lc) {
      //multiplicity of the current LC
      float mlp = std::count(std::begin(multiplicity[tstId]), std::end(multiplicity[tstId]), multiplicity[tstId][lc]);
      //LogDebug("HGCalValidator") << "mlp %" << (100. * mlp)/ ((float) nLayerClusters) << std::endl;
      // histograms.h_multiplicityOfLCinTST[count]->Fill( mlp , multiplicity[tstId][lc] , 100. / (float) totalLcInTsts );
      histograms.h_multiplicityOfLCinTST[count]->Fill(mlp, multiplicity[tstId][lc]);
      //When plotting with the text option we want the entries to be the same
      //as the % of the current cell over the whole number of layerClusters. For this we need an extra histo.
      histograms.h_multiplicity_numberOfEventsHistogram[count]->Fill(mlp);
      //For the cluster multiplicity vs layer
      //First with the -z endcap (V10:0->49)
      if (multiplicity_vs_layer[tstId][lc] < layers) {
        histograms.h_multiplicityOfLCinTST_vs_layercluster_zminus[count]->Fill(mlp, multiplicity_vs_layer[tstId][lc]);
        histograms.h_multiplicity_zminus_numberOfEventsHistogram[count]->Fill(mlp);
      } else {  //Then for the +z (V10:50->99)
        histograms.h_multiplicityOfLCinTST_vs_layercluster_zplus[count]->Fill(
            mlp, multiplicity_vs_layer[tstId][lc] - layers);
        histograms.h_multiplicity_zplus_numberOfEventsHistogram[count]->Fill(mlp);
      }
      //For the cluster multiplicity vs cluster energy
      histograms.h_multiplicityOfLCinTST_vs_layerclusterenergy[count]->Fill(mlp,
                                                                            layerClusters[tst.vertices(lc)].energy());
    }

    if (!trackster_layers.empty()) {
      histograms.h_trackster_x[count]->Fill(tst.barycenter().x());
      histograms.h_trackster_y[count]->Fill(tst.barycenter().y());
      histograms.h_trackster_z[count]->Fill(tst.barycenter().z());
      histograms.h_trackster_eta[count]->Fill(tst.barycenter().eta());
      histograms.h_trackster_phi[count]->Fill(tst.barycenter().phi());

      histograms.h_trackster_firstlayer[count]->Fill((float)*trackster_layers.begin());
      histograms.h_trackster_lastlayer[count]->Fill((float)*trackster_layers.rbegin());
      histograms.h_trackster_layersnum[count]->Fill((float)trackster_layers.size());

      histograms.h_trackster_pt[count]->Fill(tst.raw_pt());
      histograms.h_trackster_energy[count]->Fill(tst.raw_energy());
    }

  }  //end of loop through Tracksters

  histograms.h_tracksternum[count]->Fill(totNTstZm + totNTstZp);
  histograms.h_conttracksternum[count]->Fill(totNContTstZp + totNContTstZm);
  histograms.h_nonconttracksternum[count]->Fill(totNNotContTstZp + totNNotContTstZm);

  // CaloParticle
  tracksters_to_SimTracksters(histograms,
                              count,
                              tracksters,
                              layerClusters,
                              simTSs_fromCP,
                              Linking,
                              simTSs_fromCP,
                              cpToSc_SimTrackstersMap,
                              sC,
                              cPHandle_id,
                              cP,
                              cPIndices,
                              cPSelectedIndices,
                              hitMap,
                              layers);

  // Pattern recognition
  tracksters_to_SimTracksters(histograms,
                              count,
                              tracksters,
                              layerClusters,
                              simTSs,
                              PatternRecognition,
                              simTSs_fromCP,
                              cpToSc_SimTrackstersMap,
                              sC,
                              cPHandle_id,
                              cP,
                              cPIndices,
                              cPSelectedIndices,
                              hitMap,
                              layers);
}

double HGVHistoProducerAlgo::distance2(const double x1,
                                       const double y1,
                                       const double x2,
                                       const double y2) const {  //distance squared
  const double dx = x1 - x2;
  const double dy = y1 - y2;
  return (dx * dx + dy * dy);
}  //distance squaredq
double HGVHistoProducerAlgo::distance(const double x1,
                                      const double y1,
                                      const double x2,
                                      const double y2) const {  //2-d distance on the layer (x-y)
  return std::sqrt(distance2(x1, y1, x2, y2));
}

void HGVHistoProducerAlgo::setRecHitTools(std::shared_ptr<hgcal::RecHitTools> recHitTools) {
  recHitTools_ = recHitTools;
}

DetId HGVHistoProducerAlgo::findmaxhit(const reco::CaloCluster& cluster,
                                       std::unordered_map<DetId, const HGCRecHit*> const& hitMap) const {
  const auto& hits_and_fractions = cluster.hitsAndFractions();

  DetId themaxid;
  double maxene = 0.;
  for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin();
       it_haf != hits_and_fractions.end();
       ++it_haf) {
    const DetId rh_detid = it_haf->first;
    const auto hitEn = hitMap.find(rh_detid)->second->energy();

    if (maxene < hitEn) {
      maxene = hitEn;
      themaxid = rh_detid;
    }
  }

  return themaxid;
}

double HGVHistoProducerAlgo::getEta(double eta) const {
  if (useFabsEta_)
    return fabs(eta);
  else
    return eta;
}
