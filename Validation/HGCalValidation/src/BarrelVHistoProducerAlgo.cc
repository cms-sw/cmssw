#include <numeric>
#include <iomanip>
#include <sstream>

#include "Validation/HGCalValidation/interface/BarrelVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "TMath.h"
#include "TLatex.h"
#include "TF1.h"

using namespace std;

//Parameters for the score cut. Later, this will become part of the
//configuration parameter for the Barrel associator.
const double ScoreCutLCtoCP_ = 0.1;
const double ScoreCutCPtoLC_ = 0.1;
const double ScoreCutLCtoSC_ = 0.1;
const double ScoreCutSCtoLC_ = 0.1;

BarrelVHistoProducerAlgo::BarrelVHistoProducerAlgo(const edm::ParameterSet& pset)
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

      //parameters for the total amount of energy clustered by all layer clusters (fraction over CaloParticless)
      minEneCl_(pset.getParameter<double>("minEneCl")),
      maxEneCl_(pset.getParameter<double>("maxEneCl")),
      nintEneCl_(pset.getParameter<int>("nintEneCl")),

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

BarrelVHistoProducerAlgo::~BarrelVHistoProducerAlgo() {}

void BarrelVHistoProducerAlgo::bookInfo(DQMStore::IBooker& ibook, Histograms& histograms) {
  histograms.lastLayerEB = ibook.bookInt("lastLayerEB");
  histograms.lastLayerHB = ibook.bookInt("lastLayerHB");
}

void BarrelVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::IBooker& ibook,
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
                   layers,
                   0.,
                   (float)layers,
                   100,
                   0.,
                   10.);
  histograms.h_caloparticle_nHits_matched_energy_layer_1SimCl[pdgid] =
      ibook.book2D("Energy of Rec-matched Hits vs layer (1SC)",
                   "Energy of Hits only 1 Sim Clusters (matched) vs layer",
                   layers,
                   0.,
                   (float)layers,
                   100,
                   0.,
                   10.);
  histograms.h_caloparticle_sum_energy_layer[pdgid] =
      ibook.book2D("Rec-matched Hits Sum Energy vs layer",
                   "Rescaled Sum Energy of Hits in Sim Clusters (matched) vs layer",
                   layers,
                   0.,
                   (float)layers,
                   110,
                   0.,
                   110.);
  histograms.h_caloparticle_fractions[pdgid] =
      ibook.book2D("HitFractions", "Hit fractions;Hit fraction;E_{hit}^{2} fraction", 101, 0, 1.01, 100, 0, 1);
  histograms.h_caloparticle_fractions_weight[pdgid] = ibook.book2D(
      "HitFractions_weighted", "Hit fractions weighted;Hit fraction;E_{hit}^{2} fraction", 101, 0, 1.01, 100, 0, 1);

  histograms.h_caloparticle_firstlayer[pdgid] =
      ibook.book1D("First Layer", "First layer of the CaloParticles", layers, 0., (float)layers);
  histograms.h_caloparticle_lastlayer[pdgid] =
      ibook.book1D("Last Layer", "Last layer of the CaloParticles", layers, 0., (float)layers);
  histograms.h_caloparticle_layersnum[pdgid] =
      ibook.book1D("Number of Layers", "Number of layers of the CaloParticles", layers, 0., (float)layers);
  histograms.h_caloparticle_firstlayer_matchedtoRecHit[pdgid] = ibook.book1D(
      "First Layer (rec-matched hit)", "First layer of the CaloParticles (matched)", layers, 0., (float)layers);
  histograms.h_caloparticle_lastlayer_matchedtoRecHit[pdgid] = ibook.book1D(
      "Last Layer (rec-matched hit)", "Last layer of the CaloParticles (matched)", layers, 0., (float)layers);
  histograms.h_caloparticle_layersnum_matchedtoRecHit[pdgid] =
      ibook.book1D("Number of Layers (rec-matched hit)",
                   "Number of layers of the CaloParticles (matched)",
                   layers,
                   0.,
                   (float)layers);
}

void BarrelVHistoProducerAlgo::bookSimClusterHistos(DQMStore::IBooker& ibook,
                                                    Histograms& histograms,
                                                    unsigned int layers) {
  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    std::string istr2 = istr1 + " in barrel";
    istr1 = "_barrel" + istr1;
    histograms.h_simclusternum_perlayer[ilayer] = ibook.book1D("totsimclusternum_layer_" + istr1,
                                                               "total number of SimClusters for layer " + istr1,
                                                               nintTotNsimClsperlay_,
                                                               minTotNsimClsperlay_,
                                                               maxTotNsimClsperlay_);

  }  //end of loop over layers
}

void BarrelVHistoProducerAlgo::bookSimClusterAssociationHistos(DQMStore::IBooker& ibook,
                                                               Histograms& histograms,
                                                               unsigned int layers) {
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
  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    std::string istr2 = istr1 + " in barrel";
    istr1 = "_barrel" + istr1;
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
void BarrelVHistoProducerAlgo::bookClusterHistos_ClusterLevel(DQMStore::IBooker& ibook,
                                                              Histograms& histograms,
                                                              unsigned int layers) {
  //---------------------------------------------------------------------------------------------------------------------------
  histograms.h_cluster_eta.push_back(
      ibook.book1D("num_reco_cluster_eta", "N of reco clusters vs eta", nintEta_, minEta_, maxEta_));

  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    std::string istr2 = istr1 + " in barrel";
    istr1 = "_barrel" + istr1;

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
}

void BarrelVHistoProducerAlgo::bookClusterHistos_LCtoCP_association(DQMStore::IBooker& ibook,
                                                                    Histograms& histograms,
                                                                    unsigned int layers) {
  //----------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    std::string istr2 = istr1 + " in barrel";
    istr1 = "_barrel" + istr1;
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

void BarrelVHistoProducerAlgo::bookClusterHistos_CellLevel(DQMStore::IBooker& ibook,
                                                           Histograms& histograms,
                                                           unsigned int layers) {
  //----------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    std::string istr2 = istr1 + " in barrel";
    istr1 = "_barrel" + istr1;
    histograms.h_cellAssociation_perlayer[ilayer] =
        ibook.book1D("cellAssociation_perlayer" + istr1, "Cell Association for layer " + istr2, 5, -4., 1.);
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(2, "TN(purity)");
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(3, "FN(ineff.)");
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(4, "FP(fake)");
    histograms.h_cellAssociation_perlayer[ilayer]->setBinLabel(5, "TP(eff.)");
  }
}
//----------------------------------------------------------------------------------------------------------------------------

void BarrelVHistoProducerAlgo::fill_info_histos(const Histograms& histograms, unsigned int layers) const {
  // Save some info straight from geometry to avoid mistakes from updates
  //----------- TODO ----------------------------------------------------------
  // For now values returned for 'lastLayerFHzp': '104', 'lastLayerFHzm': '52' are not the one expected.
  // Will come back to this when there will be info in CMSSW to put in DQM file.
  histograms.lastLayerEB->Fill(0);
  histograms.lastLayerHB->Fill(recHitTools_->lastLayerBarrel());
}

void BarrelVHistoProducerAlgo::fill_caloparticle_histos(
    const Histograms& histograms,
    int pdgid,
    const CaloParticle& caloParticle,
    std::vector<SimVertex> const& simVertices,
    unsigned int layers,
    std::unordered_map<DetId, const unsigned int> const& barrelHitMap,
    edm::MultiSpan<reco::PFRecHit> const& barrelHits) const {
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
      auto barrel_hf = sc->filtered_hits_and_fractions([this](const DetId& x) { return recHitTools_->isBarrel(x); });

      LogDebug("BarrelValidator") << " This sim cluster has " << barrel_hf.size() << " simHits and " << sc->energy()
                                  << " energy. " << std::endl;

      simHits += barrel_hf.size();
      for (auto const& h_and_f : barrel_hf) {
        const auto hitDetId = h_and_f.first;
        const int layerId = recHitTools_->getLayerWithOffset(hitDetId);
        // set to 0 if matched RecHit not found
        int layerId_matched_min = 999;
        int layerId_matched_max = 0;
        std::unordered_map<DetId, const unsigned int>::const_iterator itcheck = barrelHitMap.find(hitDetId);
        if (itcheck != barrelHitMap.end()) {
          layerId_matched_min = layerId;
          layerId_matched_max = layerId;
          simHits_matched++;

          const auto hitEn = (barrelHits[itcheck->second]).energy();
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
          LogDebug("BarrelValidator") << "   matched to RecHit NOT found !" << std::endl;
        }

        minLayerId = std::min(minLayerId, layerId);
        maxLayerId = std::max(maxLayerId, layerId);
        minLayerId_matched = std::min(minLayerId_matched, layerId_matched_min);
        maxLayerId_matched = std::max(maxLayerId_matched, layerId_matched_max);
      }
      LogDebug("BarrelValidator") << std::endl;
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
      if (!recHitTools_->isBarrel(haf.first))
        continue;
      const auto hitEn = (barrelHits[barrelHitMap.find(haf.first)->second]).energy();
      const auto weight = pow(hitEn, 2);
      histograms.h_caloparticle_fractions.at(pdgid)->Fill(haf.second, weight * hitEnergyWeight_invSum);
      histograms.h_caloparticle_fractions_weight.at(pdgid)->Fill(haf.second, weight * hitEnergyWeight_invSum, weight);
    }
  }
}

void BarrelVHistoProducerAlgo::BarrelVHistoProducerAlgo::fill_simCluster_histos(
    const Histograms& histograms, std::vector<SimCluster> const& simClusters, unsigned int layers) const {
  //To keep track of total num of simClusters per layer
  std::vector<int> tnscpl(layers, 0);

  //loop through simClusters
  for (const auto& sc : simClusters) {
    //To keep track if we added the simCluster in a specific layer
    std::vector<int> occurenceSCinlayer(layers, 0);

    //loop through hits of the simCluster
    for (const auto& hAndF :
         sc.filtered_hits_and_fractions([this](const DetId& x) { return recHitTools_->isBarrel(x); })) {
      const DetId sh_detid = hAndF.first;

      //The layer the cluster belongs to
      int layerid = recHitTools_->getLayerWithOffset(sh_detid);
      if (occurenceSCinlayer[layerid] == 0) {
        tnscpl[layerid]++;
      }
      occurenceSCinlayer[layerid]++;
    }  //end of loop through hits
  }  //end of loop through SimClusters of the event

  //Per layer : Loop 0->99
  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    if (histograms.h_simclusternum_perlayer.count(ilayer))
      histograms.h_simclusternum_perlayer.at(ilayer)->Fill(tnscpl[ilayer]);
  }
}

void BarrelVHistoProducerAlgo::BarrelVHistoProducerAlgo::fill_simClusterAssociation_histos(
    const Histograms& histograms,
    const int count,
    edm::Handle<reco::CaloClusterCollection> clusterHandle,
    const reco::CaloClusterCollection& clusters,
    edm::Handle<std::vector<SimCluster>> simClusterHandle,
    std::vector<SimCluster> const& simClusters,
    std::vector<size_t> const& sCIndices,
    const std::vector<float>& mask,
    std::unordered_map<DetId, const unsigned int> const& barrelHitMap,
    unsigned int layers,
    const ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection>& scsInLayerClusterMap,
    const ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection>& lcsInSimClusterMap,
    edm::MultiSpan<reco::PFRecHit> const& barrelHits) const {
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
                               barrelHitMap,
                               layers,
                               scsInLayerClusterMap,
                               lcsInSimClusterMap,
                               barrelHits);
}

void BarrelVHistoProducerAlgo::fill_cluster_histos(const Histograms& histograms,
                                                   const int count,
                                                   const reco::CaloCluster& cluster) const {
  if (!recHitTools_->isBarrel(cluster.seed()))
    return;
  const auto eta = getEta(cluster.eta());
  histograms.h_cluster_eta[count]->Fill(eta);
}

void BarrelVHistoProducerAlgo::layerClusters_to_CaloParticles(
    const Histograms& histograms,
    edm::Handle<reco::CaloClusterCollection> clusterHandle,
    const reco::CaloClusterCollection& clusters,
    edm::Handle<std::vector<CaloParticle>> caloParticleHandle,
    std::vector<CaloParticle> const& cP,
    std::vector<size_t> const& cPIndices,
    std::vector<size_t> const& cPSelectedIndices,
    std::unordered_map<DetId, const unsigned int> const& barrelHitMap,
    unsigned int layers,
    const ticl::RecoToSimCollectionT<reco::CaloClusterCollection>& cpsInLayerClusterMap,
    const ticl::SimToRecoCollectionT<reco::CaloClusterCollection>& cPOnLayerMap,
    edm::MultiSpan<reco::PFRecHit> const& barrelHits) const {
  const auto nLayerClusters = clusters.size();

  std::unordered_map<DetId, std::vector<BarrelVHistoProducerAlgo::detIdInfoInCluster>> detIdToCaloParticleId_Map;
  std::unordered_map<DetId, std::vector<BarrelVHistoProducerAlgo::detIdInfoInCluster>> detIdToLayerClusterId_Map;

  // The association has to be done in an all-vs-all fashion.
  // For this reason use the full set of CaloParticles, with the only filter on bx
  for (const auto& cpId : cPIndices) {
    for (const auto& simCluster : cP[cpId].simClusters()) {
      for (const auto& it_haf : simCluster->hits_and_fractions()) {
        const DetId hitid = (it_haf.first);
        if (barrelHitMap.find(hitid) != barrelHitMap.end()) {
          if (detIdToCaloParticleId_Map.find(hitid) == detIdToCaloParticleId_Map.end()) {
            detIdToCaloParticleId_Map[hitid] = std::vector<BarrelVHistoProducerAlgo::detIdInfoInCluster>();
            detIdToCaloParticleId_Map[hitid].emplace_back(
                BarrelVHistoProducerAlgo::detIdInfoInCluster{cpId, it_haf.second});
          } else {
            auto findHitIt =
                std::find(detIdToCaloParticleId_Map[hitid].begin(),
                          detIdToCaloParticleId_Map[hitid].end(),
                          BarrelVHistoProducerAlgo::detIdInfoInCluster{
                              cpId, 0.f});  // only the first element is used for the matching (overloaded operator==)
            if (findHitIt != detIdToCaloParticleId_Map[hitid].end())
              findHitIt->fraction += it_haf.second;
            else
              detIdToCaloParticleId_Map[hitid].emplace_back(
                  BarrelVHistoProducerAlgo::detIdInfoInCluster{cpId, it_haf.second});
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
    int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId);
    bool isBarrel = recHitTools_->isBarrel(firstHitDetId);
    if (!isBarrel)
      continue;
    // This will store the fraction of the CaloParticle energy shared with the LayerCluster: e_shared/cp_energy
    std::unordered_map<unsigned, float> CPEnergyInLC;

    for (unsigned int iHit = 0; iHit < numberOfHitsInLC; iHit++) {
      const DetId rh_detid = hits_and_fractions[iHit].first;
      const auto rhFraction = hits_and_fractions[iHit].second;

      float hit_energy;
      if (!isBarrel) {
        continue;
      } else {
        std::unordered_map<DetId, const unsigned int>::const_iterator itcheck = barrelHitMap.find(rh_detid);
        const reco::PFRecHit* hit = &(barrelHits[itcheck->second]);
        hit_energy = hit->energy();
      }
      if (detIdToLayerClusterId_Map.find(rh_detid) == detIdToLayerClusterId_Map.end()) {
        detIdToLayerClusterId_Map[rh_detid] = std::vector<BarrelVHistoProducerAlgo::detIdInfoInCluster>();
      }
      detIdToLayerClusterId_Map[rh_detid].emplace_back(BarrelVHistoProducerAlgo::detIdInfoInCluster{lcId, rhFraction});

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
          CPEnergyInLC[iCP] += h.fraction * hit_energy;
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
  // restrict only to the selected caloParticles.
  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId) {
    const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
    bool isBarrel = recHitTools_->isBarrel(firstHitDetId);
    int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId);
    if (!isBarrel)
      continue;
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
      LogDebug("BarrelValidator") << "layerCluster Id: \t" << lcId << "\t CP id: \t" << cpPair.first.index()
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
    for (unsigned int layerId = 0; layerId < layers; ++layerId)
      cPEnergyOnLayer[layerId] = 0;

    for (const auto& simCluster : cP[cpId].simClusters()) {
      for (const auto& it_haf : simCluster->hits_and_fractions()) {
        const DetId hitid = (it_haf.first);
        bool isBarrel = recHitTools_->isBarrel(hitid);
        auto hitLayerId = recHitTools_->getLayerWithOffset(hitid);
        if (!isBarrel) {
          continue;
        } else {
          std::unordered_map<DetId, const unsigned int>::const_iterator itcheck = barrelHitMap.find(hitid);
          if (itcheck != barrelHitMap.end()) {
            const reco::PFRecHit* hit = &(barrelHits[itcheck->second]);
            cPEnergyOnLayer[hitLayerId] += it_haf.second * hit->energy();
          }
        }
      }
    }

    for (unsigned int layerId = 0; layerId < layers; ++layerId) {
      if (!cPEnergyOnLayer[layerId])
        continue;

      histograms.h_denom_caloparticle_eta_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().eta());
      histograms.h_denom_caloparticle_phi_perlayer.at(layerId)->Fill(cP[cpId].g4Tracks()[0].momentum().phi());

      if (lcsIt == cPOnLayerMap.end())
        continue;
      const auto& lcs = lcsIt->val;

      auto getLCLayerId = [&](const unsigned int lcId) {
        const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
        bool isBarrel = recHitTools_->isBarrel(firstHitDetId);
        if (!isBarrel)
          return 9999u;
        auto lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId);
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

void BarrelVHistoProducerAlgo::layerClusters_to_SimClusters(
    const Histograms& histograms,
    const int count,
    edm::Handle<reco::CaloClusterCollection> clusterHandle,
    const reco::CaloClusterCollection& clusters,
    edm::Handle<std::vector<SimCluster>> simClusterHandle,
    std::vector<SimCluster> const& sC,
    std::vector<size_t> const& sCIndices,
    const std::vector<float>& mask,
    std::unordered_map<DetId, const unsigned int> const& barrelHitMap,
    unsigned int layers,
    const ticl::RecoToSimCollectionWithSimClustersT<reco::CaloClusterCollection>& scsInLayerClusterMap,
    const ticl::SimToRecoCollectionWithSimClustersT<reco::CaloClusterCollection>& lcsInSimClusterMap,
    edm::MultiSpan<reco::PFRecHit> const& barrelHits) const {
  // Here fill the plots to compute the different metrics linked to
  // reco-level, namely fake-rate and merge-rate. In this loop should *not*
  // restrict only to the selected SimClusters.
  for (unsigned int lcId = 0; lcId < clusters.size(); ++lcId) {
    if (mask[lcId] != 0.) {
      LogDebug("BarrelValidator") << "Skipping layer cluster " << lcId << " not belonging to mask" << std::endl;
      continue;
    }
    const auto firstHitDetId = (clusters[lcId].hitsAndFractions())[0].first;
    bool isBarrel = recHitTools_->isBarrel(firstHitDetId);
    auto lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId);
    if (!isBarrel)
      continue;
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
      LogDebug("BarrelValidator") << "layerCluster Id: \t" << lcId << "\t SC id: \t" << scPair.first.index()
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
      bool isBarrel = recHitTools_->isBarrel(hitid);
      auto scLayerId = recHitTools_->getLayerWithOffset(hitid);
      if (!isBarrel) {
        continue;
      } else {
        std::unordered_map<DetId, const unsigned int>::const_iterator itcheck = barrelHitMap.find(hitid);
        if (itcheck != barrelHitMap.end()) {
          const reco::PFRecHit* hit = &(barrelHits[itcheck->second]);
          sCEnergyOnLayer[scLayerId] += it_haf.second * hit->energy();
        }
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
        bool isBarrel = recHitTools_->isBarrel(firstHitDetId);
        if (!isBarrel)
          return 9999u;
        unsigned int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId);
        return lcLayerId;
      };

      //Loop through layer clusters linked to the SimCluster under study
      for (const auto& lcPair : lcs) {
        auto lcId = lcPair.first.index();
        if (mask[lcId] != 0.) {
          LogDebug("BarrelValidator") << "Skipping layer cluster " << lcId << " not belonging to mask" << std::endl;
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

void BarrelVHistoProducerAlgo::fill_generic_cluster_histos(
    const Histograms& histograms,
    const int count,
    edm::Handle<reco::CaloClusterCollection> clusterHandle,
    const reco::CaloClusterCollection& clusters,
    edm::Handle<std::vector<CaloParticle>> caloParticleHandle,
    std::vector<CaloParticle> const& cP,
    std::vector<size_t> const& cPIndices,
    std::vector<size_t> const& cPSelectedIndices,
    std::unordered_map<DetId, const unsigned int> const& barrelHitMap,
    unsigned int layers,
    const ticl::RecoToSimCollectionT<reco::CaloClusterCollection>& cpsInLayerClusterMap,
    const ticl::SimToRecoCollectionT<reco::CaloClusterCollection>& cPOnLayerMap,
    edm::MultiSpan<reco::PFRecHit> const& barrelHits) const {
  //To keep track of total num of layer clusters per layer
  std::vector<int> tnlcpl(layers, 0);

  layerClusters_to_CaloParticles(histograms,
                                 clusterHandle,
                                 clusters,
                                 caloParticleHandle,
                                 cP,
                                 cPIndices,
                                 cPSelectedIndices,
                                 barrelHitMap,
                                 layers,
                                 cpsInLayerClusterMap,
                                 cPOnLayerMap,
                                 barrelHits);

  std::vector<double> tecpl(layers, 0.0);

  // loop through clusters of the event
  for (const auto& lcId : clusters) {
    const auto seedid = lcId.seed();
    if (!recHitTools_->isBarrel(seedid)) {
      continue;
    }
    const float lc_en = lcId.energy();
    int layerid = recHitTools_->getLayerWithOffset(seedid);
    //Energy clustered per layer
    tecpl[layerid] += lc_en;
    tnlcpl[layerid] += 1;

  }  //end of loop through clusters of the event

  for (unsigned ilayer = 0; ilayer < layers; ++ilayer) {
    if (histograms.h_clusternum_perlayer.count(ilayer)) {
      histograms.h_clusternum_perlayer.at(ilayer)->Fill(tnlcpl[ilayer]);
      histograms.h_energyclustered_perlayer.at(ilayer)->Fill(tecpl[ilayer]);
    }
  }  //end of loop over layers
}

double BarrelVHistoProducerAlgo::distance2(const double x1,
                                           const double y1,
                                           const double x2,
                                           const double y2) const {  //distance squared
  const double dx = x1 - x2;
  const double dy = y1 - y2;
  return (dx * dx + dy * dy);
}  //distance squaredq
double BarrelVHistoProducerAlgo::distance(const double x1,
                                          const double y1,
                                          const double x2,
                                          const double y2) const {  //2-d distance on the layer (x-y)
  return std::sqrt(distance2(x1, y1, x2, y2));
}

void BarrelVHistoProducerAlgo::setRecHitTools(std::shared_ptr<hgcal::RecHitTools> recHitTools) {
  recHitTools_ = recHitTools;
}

DetId BarrelVHistoProducerAlgo::findmaxhit(const reco::CaloCluster& cluster,
                                           std::unordered_map<DetId, const unsigned int> const& hitMap,
                                           edm::MultiSpan<reco::PFRecHit> const& hits) const {
  const auto& hits_and_fractions = cluster.hitsAndFractions();

  DetId themaxid;
  double maxene = 0.;
  for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin();
       it_haf != hits_and_fractions.end();
       ++it_haf) {
    const DetId rh_detid = it_haf->first;
    if (!recHitTools_->isBarrel(rh_detid))
      continue;
    const auto hitEn = (hits[hitMap.find(rh_detid)->second]).energy();
    if (maxene < hitEn) {
      maxene = hitEn;
      themaxid = rh_detid;
    }
  }

  return themaxid;
}

double BarrelVHistoProducerAlgo::getEta(double eta) const {
  if (useFabsEta_)
    return fabs(eta);
  else
    return eta;
}
