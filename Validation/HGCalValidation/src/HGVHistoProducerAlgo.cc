#include <numeric>
#include <iomanip>

#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "TMath.h"
#include "TLatex.h"
#include "TF1.h"

using namespace std;

//Parameters for the score cut. Later, this will become part of the 
//configuration parameter for the HGCAL associator.
const double ScoreCutLCtoCP_ = 0.2;
const double ScoreCutCPtoLC_ = 0.2;

HGVHistoProducerAlgo::HGVHistoProducerAlgo(const edm::ParameterSet& pset) :
  //parameters for eta
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

  //parameters for counting mixed hits clusters
  minMixedHitsCluster_(pset.getParameter<double>("minMixedHitsCluster")),
  maxMixedHitsCluster_(pset.getParameter<double>("maxMixedHitsCluster")),
  nintMixedHitsCluster_(pset.getParameter<int>("nintMixedHitsCluster")),

  //parameters for the total amount of energy clustered by all layer clusters (fraction over caloparticles)
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

  //Parameters for the total number of layer clusters per layer
  minTotNClsperlay_(pset.getParameter<double>("minTotNClsperlay")),
  maxTotNClsperlay_(pset.getParameter<double>("maxTotNClsperlay")),
  nintTotNClsperlay_(pset.getParameter<int>("nintTotNClsperlay")),

  //Parameters for the energy clustered by layer clusters per layer (fraction over caloparticles)
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
  nintCellsEneDensperthick_(pset.getParameter<int>("nintCellsEneDensperthick"))

{}

HGVHistoProducerAlgo::~HGVHistoProducerAlgo() {}

void HGVHistoProducerAlgo::bookInfo(DQMStore::ConcurrentBooker& ibook, Histograms& histograms) {

  histograms.lastLayerEEzm = ibook.bookInt("lastLayerEEzm");
  histograms.lastLayerFHzm = ibook.bookInt("lastLayerFHzm");
  histograms.maxlayerzm    = ibook.bookInt("maxlayerzm"); 
  histograms.lastLayerEEzp = ibook.bookInt("lastLayerEEzp"); 
  histograms.lastLayerFHzp = ibook.bookInt("lastLayerFHzp"); 
  histograms.maxlayerzp    = ibook.bookInt("maxlayerzp"); 

}

void HGVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid) {

  histograms.h_caloparticle_eta[pdgid] = ibook.book1D("num_caloparticle_eta","N of caloparticle vs eta",nintEta_,minEta_,maxEta_);
  histograms.h_caloparticle_eta_Zorigin[pdgid] = ibook.book2D("Eta vs Zorigin", "Eta vs Zorigin", nintEta_, minEta_, maxEta_, nintZpos_, minZpos_, maxZpos_);

  histograms.h_caloparticle_energy[pdgid] = ibook.book1D("caloparticle_energy", "Energy of caloparticle", nintEne_,minEne_,maxEne_);
  histograms.h_caloparticle_pt[pdgid] = ibook.book1D("caloparticle_pt", "Pt of caloparticle", nintPt_,minPt_,maxPt_);
  histograms.h_caloparticle_phi[pdgid] = ibook.book1D("caloparticle_phi", "Phi of caloparticle", nintPhi_,minPhi_,maxPhi_);


}


void HGVHistoProducerAlgo::bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms, unsigned layers, std::vector<int> thicknesses, std::string pathtomatbudfile) {

  //---------------------------------------------------------------------------------------------------------------------------
  histograms.h_cluster_eta.push_back( ibook.book1D("num_reco_cluster_eta","N of reco clusters vs eta",nintEta_,minEta_,maxEta_) );

  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_mixedhitscluster_zminus.push_back( ibook.book1D("mixedhitscluster_zminus","N of reco clusters that contain hits of more than one kind in z-",nintMixedHitsCluster_,minMixedHitsCluster_,maxMixedHitsCluster_) );
  //z+
  histograms.h_mixedhitscluster_zplus.push_back( ibook.book1D("mixedhitscluster_zplus","N of reco clusters that contain hits of more than one kind in z+",nintMixedHitsCluster_,minMixedHitsCluster_,maxMixedHitsCluster_) );
  
  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_energyclustered_zminus.push_back( ibook.book1D("energyclustered_zminus","percent of total energy clustered by all layer clusters over caloparticles energy in z-",nintEneCl_,minEneCl_,maxEneCl_) );
  //z+
  histograms.h_energyclustered_zplus.push_back( ibook.book1D("energyclustered_zplus","percent of total energy clustered by all layer clusters over caloparticles energy in z+",nintEneCl_,minEneCl_,maxEneCl_) );
  
  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  std::string subpathtomat = pathtomatbudfile.substr(pathtomatbudfile.find("Validation"));
  histograms.h_longdepthbarycentre_zminus.push_back( ibook.book1D("longdepthbarycentre_zminus","The longitudinal depth barycentre in z- for "+subpathtomat,nintLongDepBary_,minLongDepBary_,maxLongDepBary_) );
  //z+
  histograms.h_longdepthbarycentre_zplus.push_back( ibook.book1D("longdepthbarycentre_zplus","The longitudinal depth barycentre in z+ for "+subpathtomat,nintLongDepBary_,minLongDepBary_,maxLongDepBary_) );


  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2*layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    while(istr1.size() < 2) {istr1.insert(0, "0");}
    //We will make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
    //First with the -z endcap
    if (ilayer < layers){
      istr2 = std::to_string(ilayer + 1) + " in z-";
    }else { //Then for the +z
      istr2 = std::to_string(ilayer - 51) + " in z+";
    }
    histograms.h_clusternum_perlayer[ilayer] = ibook.book1D("totclusternum_layer_"+istr1,"total number of layer clusters for layer "+istr2,nintTotNClsperlay_,minTotNClsperlay_,maxTotNClsperlay_);
    histograms.h_energyclustered_perlayer[ilayer] = ibook.book1D("energyclustered_perlayer"+istr1,"percent of total energy clustered by layer clusters over caloparticles energy for layer "+istr2,nintEneClperlay_,minEneClperlay_,maxEneClperlay_);
    histograms.h_score_layercl2caloparticle_perlayer[ilayer] = ibook.book1D("Score_layercl2caloparticle_perlayer"+istr1, "Score of Layer Cluster per CaloParticle for layer "+istr2, nintScore_,minScore_,maxScore_); 
    histograms.h_score_caloparticle2layercl_perlayer[ilayer] = ibook.book1D("Score_caloparticle2layercl_perlayer"+istr1, "Score of CaloParticle per Layer Cluster for layer "+istr2, nintScore_,minScore_,maxScore_);
    histograms.h_energy_vs_score_caloparticle2layercl_perlayer[ilayer] = ibook.book2D("Energy_vs_Score_caloparticle2layer_perlayer"+istr1, "Energy vs Score of CaloParticle per Layer Cluster for layer "+istr2, nintScore_,minScore_,maxScore_, nintSharedEneFrac_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_energy_vs_score_layercl2caloparticle_perlayer[ilayer] = ibook.book2D("Energy_vs_Score_layer2caloparticle_perlayer"+istr1, "Energy vs Score of Layer Cluster per CaloParticle Layer for layer "+istr2, nintScore_,minScore_,maxScore_, nintSharedEneFrac_, minSharedEneFrac_, maxSharedEneFrac_); 
    histograms.h_sharedenergy_caloparticle2layercl_perlayer[ilayer] = ibook.book1D("SharedEnergy_caloparticle2layercl_perlayer"+istr1, "Shared Energy of CaloParticle per Layer Cluster for layer "+istr2, nintSharedEneFrac_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_sharedenergy_caloparticle2layercl_vs_eta_perlayer[ilayer] = ibook.bookProfile("SharedEnergy_caloparticle2layercl_vs_eta_perlayer"+istr1, "Shared Energy of CaloParticle vs #eta per best Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_sharedenergy_caloparticle2layercl_vs_phi_perlayer[ilayer] = ibook.bookProfile("SharedEnergy_caloparticle2layercl_vs_phi_perlayer"+istr1, "Shared Energy of CaloParticle vs #phi per best Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_sharedenergy_layercl2caloparticle_perlayer[ilayer] = ibook.book1D("SharedEnergy_layercluster2caloparticle_perlayer"+istr1, "Shared Energy of Layer Cluster per Layer Calo Particle for layer "+istr2, nintSharedEneFrac_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_sharedenergy_layercl2caloparticle_vs_eta_perlayer[ilayer] = ibook.bookProfile("SharedEnergy_layercl2caloparticle_vs_eta_perlayer"+istr1, "Shared Energy of LayerCluster vs #eta per best Calo Particle for layer "+istr2, nintEta_, minEta_, maxEta_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_sharedenergy_layercl2caloparticle_vs_phi_perlayer[ilayer] = ibook.bookProfile("SharedEnergy_layercl2caloparticle_vs_phi_perlayer"+istr1, "Shared Energy of LayerCluster vs #phi per best Calo Particle for layer "+istr2, nintPhi_, minPhi_, maxPhi_, minSharedEneFrac_, maxSharedEneFrac_);
    histograms.h_num_caloparticle_eta_perlayer[ilayer] = ibook.book1D("Num_CaloParticle_Eta_perlayer"+istr1, "Num CaloParticle Eta per Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_);
    histograms.h_numDup_caloparticle_eta_perlayer[ilayer] = ibook.book1D("NumDup_CaloParticle_Eta_perlayer"+istr1, "Num Duplicate CaloParticle Eta per Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_);
    histograms.h_denom_caloparticle_eta_perlayer[ilayer] = ibook.book1D("Denom_CaloParticle_Eta_perlayer"+istr1, "Denom CaloParticle Eta per Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_);
    histograms.h_num_caloparticle_phi_perlayer[ilayer] = ibook.book1D("Num_CaloParticle_Phi_perlayer"+istr1, "Num CaloParticle Phi per Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_);
    histograms.h_numDup_caloparticle_phi_perlayer[ilayer] = ibook.book1D("NumDup_CaloParticle_Phi_perlayer"+istr1, "Num Duplicate CaloParticle Phi per Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_);
    histograms.h_denom_caloparticle_phi_perlayer[ilayer] = ibook.book1D("Denom_CaloParticle_Phi_perlayer"+istr1, "Denom CaloParticle Phi per Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_);
    histograms.h_num_layercl_eta_perlayer[ilayer] = ibook.book1D("Num_LayerCluster_Eta_perlayer"+istr1, "Num LayerCluster Eta per Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_);
    histograms.h_numMerge_layercl_eta_perlayer[ilayer] = ibook.book1D("NumMerge_LayerCluster_Eta_perlayer"+istr1, "Num Merge LayerCluster Eta per Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_);
    histograms.h_denom_layercl_eta_perlayer[ilayer] = ibook.book1D("Denom_LayerCluster_Eta_perlayer"+istr1, "Denom LayerCluster Eta per Layer Cluster for layer "+istr2, nintEta_, minEta_, maxEta_);
    histograms.h_num_layercl_phi_perlayer[ilayer] = ibook.book1D("Num_LayerCluster_Phi_perlayer"+istr1, "Num LayerCluster Phi per Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_);
    histograms.h_numMerge_layercl_phi_perlayer[ilayer] = ibook.book1D("NumMerge_LayerCluster_Phi_perlayer"+istr1, "Num Merge LayerCluster Phi per Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_);
    histograms.h_denom_layercl_phi_perlayer[ilayer] = ibook.book1D("Denom_LayerCluster_Phi_perlayer"+istr1, "Denom LayerCluster Phi per Layer Cluster for layer "+istr2, nintPhi_, minPhi_, maxPhi_);
    histograms.h_cellAssociation_perlayer[ilayer] = ibook.book1D("cellAssociation_perlayer"+istr1, "Cell Association for layer "+istr2, 5, -4., 1.);
    histograms.h_cellAssociation_perlayer[ilayer].setBinLabel(2, "TN(purity)");
    histograms.h_cellAssociation_perlayer[ilayer].setBinLabel(3, "FN(ineff.)");
    histograms.h_cellAssociation_perlayer[ilayer].setBinLabel(4, "FP(fake)");
    histograms.h_cellAssociation_perlayer[ilayer].setBinLabel(5, "TP(eff.)");
  }

  //---------------------------------------------------------------------------------------------------------------------------
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    auto istr = std::to_string(*it);
    histograms.h_clusternum_perthick[(*it)] = ibook.book1D("totclusternum_thick_"+istr,"total number of layer clusters for thickness "+istr,nintTotNClsperthick_,minTotNClsperthick_,maxTotNClsperthick_);
    //---
    histograms.h_cellsenedens_perthick[(*it)] = ibook.book1D("cellsenedens_thick_"+istr,"energy density of cluster cells for thickness "+istr,nintCellsEneDensperthick_,minCellsEneDensperthick_,maxCellsEneDensperthick_);
  }

  //---------------------------------------------------------------------------------------------------------------------------
  //Not all combination exists but we should keep them all for cross checking reason.
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    for (unsigned ilayer = 0; ilayer < 2*layers; ++ilayer) {
      auto istr1 = std::to_string(*it);
      auto istr2 = std::to_string(ilayer);
      while(istr2.size() < 2)
        istr2.insert(0, "0");
      auto istr = istr1 + "_" + istr2;
      //We will make a mapping to the regural layer naming plus z- or z+ for convenience
      std::string istr3 = "";
      //First with the -z endcap
      if (ilayer < layers){
	istr3 = std::to_string(ilayer + 1) + " in z- ";
      }else { //Then for the +z
	istr3 = std::to_string(ilayer - 51) + " in z+ ";
      }
      //---
      histograms.h_cellsnum_perthickperlayer[istr] = ibook.book1D("cellsnum_perthick_perlayer_"+istr,"total number of cells for layer "+ istr3+" for thickness "+istr1,nintTotNcellsperthickperlayer_,minTotNcellsperthickperlayer_,maxTotNcellsperthickperlayer_);
      //---
      histograms.h_distancetoseedcell_perthickperlayer[istr] = ibook.book1D("distancetoseedcell_perthickperlayer_"+istr,"distance of cluster cells to seed cell for layer "+ istr3+" for thickness "+istr1,nintDisToSeedperthickperlayer_,minDisToSeedperthickperlayer_,maxDisToSeedperthickperlayer_);
      //---
      histograms.h_distancetoseedcell_perthickperlayer_eneweighted[istr] = ibook.book1D("distancetoseedcell_perthickperlayer_eneweighted_"+istr,"energy weighted distance of cluster cells to seed cell for layer "+ istr3+" for thickness "+istr1,nintDisToSeedperthickperlayerenewei_,minDisToSeedperthickperlayerenewei_,maxDisToSeedperthickperlayerenewei_);
      //---
      histograms.h_distancetomaxcell_perthickperlayer[istr] = ibook.book1D("distancetomaxcell_perthickperlayer_"+istr,"distance of cluster cells to max cell for layer "+ istr3+" for thickness "+istr1,nintDisToMaxperthickperlayer_,minDisToMaxperthickperlayer_,maxDisToMaxperthickperlayer_);
      //---
      histograms.h_distancetomaxcell_perthickperlayer_eneweighted[istr] = ibook.book1D("distancetomaxcell_perthickperlayer_eneweighted_"+istr,"energy weighted distance of cluster cells to max cell for layer "+ istr3+" for thickness "+istr1,nintDisToMaxperthickperlayerenewei_,minDisToMaxperthickperlayerenewei_,maxDisToMaxperthickperlayerenewei_);
      //---
      histograms.h_distancebetseedandmaxcell_perthickperlayer[istr] = ibook.book1D("distancebetseedandmaxcell_perthickperlayer_"+istr,"distance of seed cell to max cell for layer "+ istr3+" for thickness "+istr1,nintDisSeedToMaxperthickperlayer_,minDisSeedToMaxperthickperlayer_,maxDisSeedToMaxperthickperlayer_);
      //---
      histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer[istr] = ibook.book2D("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_"+istr,"distance of seed cell to max cell vs cluster energy for layer "+ istr3+" for thickness "+istr1,nintDisSeedToMaxperthickperlayer_,minDisSeedToMaxperthickperlayer_,maxDisSeedToMaxperthickperlayer_,nintClEneperthickperlayer_,minClEneperthickperlayer_,maxClEneperthickperlayer_);

    }
  }
  //---------------------------------------------------------------------------------------------------------------------------

}

void HGVHistoProducerAlgo::fill_info_histos(const Histograms& histograms, unsigned layers) const {
  
  //We will save some info straight from geometry to avoid mistakes from updates
  //----------- TODO ----------------------------------------------------------
  //For now values returned for 'lastLayerFHzp': '104', 'lastLayerFHzm': '52' are not the one expected. 
  //Will come back to this when there will be info in CMSSW to put in DQM file.  
  histograms.lastLayerEEzm.fill( recHitTools_->lastLayerEE() );    
  histograms.lastLayerFHzm.fill( recHitTools_->lastLayerFH() );
  histograms.maxlayerzm.fill( layers );
  histograms.lastLayerEEzp.fill( recHitTools_->lastLayerEE() + layers );
  histograms.lastLayerFHzp.fill( recHitTools_->lastLayerFH() + layers);
  histograms.maxlayerzp.fill( layers + layers);

}


void HGVHistoProducerAlgo::fill_caloparticle_histos(const Histograms& histograms,
						    int pdgid,
						    const CaloParticle & caloparticle,
						    std::vector<SimVertex> const & simVertices) const {

  const auto eta = getEta(caloparticle.eta());
  if (histograms.h_caloparticle_eta.count(pdgid)){ histograms.h_caloparticle_eta.at(pdgid).fill(eta); }
  if (histograms.h_caloparticle_eta_Zorigin.count(pdgid)){ histograms.h_caloparticle_eta_Zorigin.at(pdgid).fill( simVertices.at(caloparticle.g4Tracks()[0].vertIndex()).position().z(), eta ); }

  if (histograms.h_caloparticle_energy.count(pdgid)){ histograms.h_caloparticle_energy.at(pdgid).fill( caloparticle.energy()  ); }
  if (histograms.h_caloparticle_pt.count(pdgid)){ histograms.h_caloparticle_pt.at(pdgid).fill( caloparticle.pt()  ); }
  if (histograms.h_caloparticle_phi.count(pdgid)){ histograms.h_caloparticle_phi.at(pdgid).fill( caloparticle.phi()  ); }


}

void HGVHistoProducerAlgo::fill_cluster_histos(const Histograms& histograms,
					       int count,
					       const reco::CaloCluster & cluster) const {

  const auto eta = getEta(cluster.eta());
  histograms.h_cluster_eta[count].fill(eta);
}

void HGVHistoProducerAlgo::layerClusters_to_CaloParticles (const Histograms& histograms,
    const reco::CaloClusterCollection &clusters,
    std::vector<CaloParticle> const & cP,
    std::map<DetId, const HGCRecHit *> const & hitMap,
    unsigned layers) const
{

  auto nLayerClusters = clusters.size();
  auto nCaloParticles = cP.size();

  std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInCluster> > detIdToCaloParticleId_Map;
  std::unordered_map<DetId, std::vector<HGVHistoProducerAlgo::detIdInfoInCluster> > detIdToLayerClusterId_Map;

  // this contains the ids of the caloparticles contributing with at least one hit to the layer cluster and the reconstruction error
  std::vector<std::vector<std::pair<unsigned int, float> > > cpsInLayerCluster;
  cpsInLayerCluster.resize(nLayerClusters);



  std::vector<std::vector<caloParticleOnLayer> > cPOnLayer;
  cPOnLayer.resize(nCaloParticles);
  for(unsigned int i = 0; i< nCaloParticles; ++i)
  {
    cPOnLayer[i].resize(layers*2);
    for(unsigned int j = 0; j< layers*2; ++j)
    {
      cPOnLayer[i][j].caloParticleId = i;
      cPOnLayer[i][j].energy = 0.f;
      cPOnLayer[i][j].hits_and_fractions.clear();
    }
  }

  for(unsigned int cpId =0; cpId < nCaloParticles; ++cpId)
  {
    const SimClusterRefVector& simClusterRefVector = cP[cpId].simClusters();
    for (const auto& it_sc : simClusterRefVector) {
      const SimCluster& simCluster = (*(it_sc));
      const auto& hits_and_fractions = simCluster.hits_and_fractions();
      for (const auto& it_haf : hits_and_fractions) {
        DetId hitid = (it_haf.first);
        int cpLayerId = recHitTools_->getLayerWithOffset(hitid) + layers * ((recHitTools_->zside(hitid) + 1) >> 1) - 1;
        std::map<DetId,const HGCRecHit *>::const_iterator itcheck= hitMap.find(hitid);
        if(itcheck != hitMap.end())
        {
          const HGCRecHit *hit = itcheck->second;
          auto hit_find_it = detIdToCaloParticleId_Map.find(hitid);
          if (hit_find_it == detIdToCaloParticleId_Map.end())
          {
            detIdToCaloParticleId_Map[hitid] = std::vector<HGVHistoProducerAlgo::detIdInfoInCluster> ();
            detIdToCaloParticleId_Map[hitid].emplace_back(HGVHistoProducerAlgo::detIdInfoInCluster{cpId,it_haf.second});
          }
          else
          {
            auto findHitIt = std::find(detIdToCaloParticleId_Map[hitid].begin(), detIdToCaloParticleId_Map[hitid].end(), HGVHistoProducerAlgo::detIdInfoInCluster{cpId,it_haf.second}) ;
            if( findHitIt != detIdToCaloParticleId_Map[hitid].end() )
            {
              findHitIt->fraction +=it_haf.second;
            }
            else
            {
              detIdToCaloParticleId_Map[hitid].emplace_back(HGVHistoProducerAlgo::detIdInfoInCluster{cpId,it_haf.second});
            }
          }
          cPOnLayer[cpId][cpLayerId].energy += it_haf.second*hit->energy();
          cPOnLayer[cpId][cpLayerId].hits_and_fractions.emplace_back(hitid,it_haf.second);
        }
      }
    }
  }


  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId)
  {
    const std::vector<std::pair<DetId, float> >& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();

    std::vector<int> hitsToCaloParticleId(numberOfHitsInLC);
    const auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId) + layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;

    int maxCPId_byNumberOfHits = -1;
    unsigned int maxCPNumberOfHitsInLC = 0;
    int maxCPId_byEnergy = -1;
    float maxEnergySharedLCandCP = 0.f;
    float energyFractionOfLCinCP = 0.f;
    float energyFractionOfCPinLC = 0.f;
    std::unordered_map<unsigned, unsigned> occurrencesCPinLC;
    std::unordered_map<unsigned, float> CPEnergyInLC;
    unsigned int numberOfNoiseHitsInLC = 0;
    unsigned int numberOfHaloHitsInLC = 0;

    for (unsigned int hitId = 0; hitId < numberOfHitsInLC; hitId++)
    {
      DetId rh_detid = hits_and_fractions[hitId].first;
      auto rhFraction = hits_and_fractions[hitId].second;

      std::map<DetId,const HGCRecHit *>::const_iterator itcheck= hitMap.find(rh_detid);
      const HGCRecHit *hit = itcheck->second;


      auto hit_find_in_LC = detIdToLayerClusterId_Map.find(rh_detid);
      if (hit_find_in_LC == detIdToLayerClusterId_Map.end())
      {
        detIdToLayerClusterId_Map[rh_detid] = std::vector<HGVHistoProducerAlgo::detIdInfoInCluster> ();
      }
      detIdToLayerClusterId_Map[rh_detid].emplace_back(HGVHistoProducerAlgo::detIdInfoInCluster{lcId,rhFraction});

      auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);

      // if the fraction is zero or the hit does not belong to any calo
      // particle, set the caloparticleId for the hit to -1 this will
      // contribute to the number of noise hits

      // MR Remove the case in which the fraction is 0, since this could be a
      // real hit that has been marked as halo.
      if (rhFraction == 0.) {
        hitsToCaloParticleId[hitId] = -2;
        numberOfHaloHitsInLC++;
      }
      if (hit_find_in_CP == detIdToCaloParticleId_Map.end())
      {
        hitsToCaloParticleId[hitId] -= 1;
      }
      else
      {
        auto maxCPEnergyInLC = 0.f;
        auto maxCPId = -1;
        for(auto& h: hit_find_in_CP->second)
        {
          CPEnergyInLC[h.clusterId] += h.fraction*hit->energy();
          cPOnLayer[h.clusterId][lcLayerId].layerClusterIdToEnergyAndScore[lcId].first += h.fraction*hit->energy();
          cpsInLayerCluster[lcId].emplace_back(std::make_pair<int, float>(h.clusterId, 0.f));
          if(h.fraction >maxCPEnergyInLC)
          {
            maxCPEnergyInLC = CPEnergyInLC[h.clusterId];
            maxCPId = h.clusterId;
          }
        }
        hitsToCaloParticleId[hitId] = maxCPId;
      }
      histograms.h_cellAssociation_perlayer.at(lcLayerId%52+1).fill(hitsToCaloParticleId[hitId] > 0. ? 0. : hitsToCaloParticleId[hitId]);
    }

    for(auto& c: hitsToCaloParticleId)
    {
      if(c < 0)
      {
        numberOfNoiseHitsInLC++;
      }
      else
      {
        occurrencesCPinLC[c]++;
      }
    }

    for(auto&c : occurrencesCPinLC)
    {
      if(c.second > maxCPNumberOfHitsInLC)
      {
        maxCPId_byNumberOfHits = c.first;
        maxCPNumberOfHitsInLC = c.second;
      }
    }

    for(auto&c : CPEnergyInLC)
    {
      if(c.second > maxEnergySharedLCandCP)
      {
        maxCPId_byEnergy = c.first;
        maxEnergySharedLCandCP = c.second;
      }
    }
    float totalCPEnergyOnLayer = 0.f;
    if(maxCPId_byEnergy >=0) {
      totalCPEnergyOnLayer = cPOnLayer[maxCPId_byEnergy][lcLayerId].energy;
      energyFractionOfCPinLC = maxEnergySharedLCandCP/totalCPEnergyOnLayer;
      if(clusters[lcId].energy()>0.f)
      {
        energyFractionOfLCinCP = maxEnergySharedLCandCP/clusters[lcId].energy();
      }
    }
    LogDebug("HGCalValidator")  << std::setw(10) << "LayerId:"<< "\t"
               << std::setw(12) << "layerCluster"<<  "\t"
               << std::setw(10) << "lc energy"<< "\t"
               << std::setw(5)  << "nhits" << "\t"
               << std::setw(12) << "noise hits" << "\t"
               << std::setw(22) << "maxCPId_byNumberOfHits" << "\t"
               << std::setw(8)  << "nhitsCP"<< "\t"
               << std::setw(16) << "maxCPId_byEnergy" << "\t"
               << std::setw(23) << "maxEnergySharedLCandCP" << "\t"
               << std::setw(22) << "totalCPEnergyOnLayer" << "\t"
               << std::setw(22) << "energyFractionOfLCinCP" << "\t"
               << std::setw(25) << "energyFractionOfCPinLC" << "\t" <<  "\n";
    LogDebug("HGCalValidator")  << std::setw(10) <<  lcLayerId << "\t"
              << std::setw(12) <<  lcId << "\t"
              << std::setw(10) <<  clusters[lcId].energy()<< "\t"
              << std::setw(5)  <<  numberOfHitsInLC << "\t"
              << std::setw(12) <<  numberOfNoiseHitsInLC << "\t"
              << std::setw(22) <<  maxCPId_byNumberOfHits << "\t"
              << std::setw(8)  <<  maxCPNumberOfHitsInLC<< "\t"
              << std::setw(16) <<  maxCPId_byEnergy << "\t"
              << std::setw(23)  <<  maxEnergySharedLCandCP << "\t"
              << std::setw(22) <<  totalCPEnergyOnLayer << "\t"
              << std::setw(22) <<  energyFractionOfLCinCP << "\t"
              << std::setw(25) <<  energyFractionOfCPinLC << "\n";
  }

  for (unsigned int lcId = 0; lcId < nLayerClusters; ++lcId)
  {
    // find the unique caloparticles id contributing to the layer clusters
    std::sort(cpsInLayerCluster[lcId].begin(), cpsInLayerCluster[lcId].end());
    auto last = std::unique(cpsInLayerCluster[lcId].begin(), cpsInLayerCluster[lcId].end());
    cpsInLayerCluster[lcId].erase(last, cpsInLayerCluster[lcId].end());
    const std::vector<std::pair<DetId, float> >& hits_and_fractions = clusters[lcId].hitsAndFractions();
    unsigned int numberOfHitsInLC = hits_and_fractions.size();
    auto firstHitDetId = hits_and_fractions[0].first;
    int lcLayerId = recHitTools_->getLayerWithOffset(firstHitDetId) + layers * ((recHitTools_->zside(firstHitDetId) + 1) >> 1) - 1;
    if (clusters[lcId].energy() == 0. && !cpsInLayerCluster[lcId].empty()) {
      for(auto& cpPair : cpsInLayerCluster[lcId]) {
        cpPair.second = 1.;
        LogDebug("HGCalValidator") << "layerCluster Id: \t" << lcId
          << "\t CP id: \t" << cpPair.first
          << "\t score \t" << cpPair.second
          << "\n";
        histograms.h_score_layercl2caloparticle_perlayer.at(lcLayerId%52+1).fill(cpPair.second);
      }
      continue;
    }
    float invLayerClusterEnergyWeight = 1.f/(clusters[lcId].energy()*clusters[lcId].energy());
    for(unsigned int i = 0; i < numberOfHitsInLC; ++i)
    {
      DetId rh_detid = hits_and_fractions[i].first;
      float rhFraction = hits_and_fractions[i].second;
      bool hitWithNoCP = false;

      auto hit_find_in_CP = detIdToCaloParticleId_Map.find(rh_detid);
      if(hit_find_in_CP == detIdToCaloParticleId_Map.end()) hitWithNoCP = true;
      auto itcheck= hitMap.find(rh_detid);
      const HGCRecHit *hit = itcheck->second;
      float hitEnergyWeight = hit->energy()*hit->energy();

      for(auto& cpPair : cpsInLayerCluster[lcId])
      {
        float cpFraction = 0.f;
        if(!hitWithNoCP)
        {
          auto findHitIt = std::find(detIdToCaloParticleId_Map[rh_detid].begin(), detIdToCaloParticleId_Map[rh_detid].end(), HGVHistoProducerAlgo::detIdInfoInCluster{cpPair.first,0.f});
          if(findHitIt != detIdToCaloParticleId_Map[rh_detid].end())
            cpFraction = findHitIt->fraction;
        }
        cpPair.second += (rhFraction - cpFraction)*(rhFraction - cpFraction)*hitEnergyWeight*invLayerClusterEnergyWeight;
      }
    }

    if(cpsInLayerCluster[lcId].empty()) LogDebug("HGCalValidator") << "layerCluster Id: \t" << lcId << "\tCP id:\t-1 " << "\t score \t-1" <<"\n";

    for(auto& cpPair : cpsInLayerCluster[lcId])
    {
      LogDebug("HGCalValidator") << "layerCluster Id: \t" << lcId
                << "\t CP id: \t" << cpPair.first
                << "\t score \t" << cpPair.second
                << "\n";
      histograms.h_score_layercl2caloparticle_perlayer.at(lcLayerId).fill(cpPair.second);
      auto const & cp_linked = cPOnLayer[cpPair.first][lcLayerId].layerClusterIdToEnergyAndScore[lcId];
      histograms.h_sharedenergy_layercl2caloparticle_perlayer.at(lcLayerId).fill(cp_linked.first/clusters[lcId].energy());
      histograms.h_energy_vs_score_layercl2caloparticle_perlayer.at(lcLayerId).fill(cpPair.second  > 1. ? 1. : cpPair.second, cp_linked.first/clusters[lcId].energy());
    }

    auto assoc = std::count_if(
        std::begin(cpsInLayerCluster[lcId]),
        std::end(cpsInLayerCluster[lcId]),
        [](const auto &obj){return obj.second < ScoreCutLCtoCP_;});
    if (assoc) {
      histograms.h_num_layercl_eta_perlayer.at(lcLayerId).fill(clusters[lcId].eta());
      histograms.h_num_layercl_phi_perlayer.at(lcLayerId).fill(clusters[lcId].phi());
      if (assoc > 1) {
        histograms.h_numMerge_layercl_eta_perlayer.at(lcLayerId).fill(clusters[lcId].eta());
        histograms.h_numMerge_layercl_phi_perlayer.at(lcLayerId).fill(clusters[lcId].phi());
      }
      auto best = std::min_element(
          std::begin(cpsInLayerCluster[lcId]),
          std::end(cpsInLayerCluster[lcId]),
          [](const auto &obj1, const auto &obj2){return obj1.second < obj2.second;});
      auto const & best_cp_linked = cPOnLayer[best->first][lcLayerId].layerClusterIdToEnergyAndScore[lcId];
      histograms.h_sharedenergy_layercl2caloparticle_vs_eta_perlayer.at(lcLayerId).fill(clusters[lcId].eta(), best_cp_linked.first/clusters[lcId].energy());
      histograms.h_sharedenergy_layercl2caloparticle_vs_phi_perlayer.at(lcLayerId).fill(clusters[lcId].phi(), best_cp_linked.first/clusters[lcId].energy());
    }
    histograms.h_denom_layercl_eta_perlayer.at(lcLayerId).fill(clusters[lcId].eta());
    histograms.h_denom_layercl_phi_perlayer.at(lcLayerId).fill(clusters[lcId].phi());
  }



  for(unsigned int cpId =0; cpId < nCaloParticles; ++cpId)
  {

    for(unsigned int layerId = 0; layerId< layers*2; ++layerId)
    {
      unsigned int CPNumberOfHits = cPOnLayer[cpId][layerId].hits_and_fractions.size();
      float CPenergy = cPOnLayer[cpId][layerId].energy;
      if(CPNumberOfHits==0) continue;
      int lcWithMaxEnergyInCP = -1;
      float maxEnergyLCinCP = 0.f;
      float CPEnergyFractionInLC = 0.f;
      for(auto& lc : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore)
      {
        if(lc.second.first > maxEnergyLCinCP)
        {
          maxEnergyLCinCP = lc.second.first;
          lcWithMaxEnergyInCP = lc.first;
        }
      }
      if(CPenergy >0.f) CPEnergyFractionInLC = maxEnergyLCinCP/CPenergy;

      LogDebug("HGCalValidator") << std::setw(8) << "LayerId:\t"
                << std::setw(12) << "caloparticle\t"
                << std::setw(15) << "cp total energy\t"
                << std::setw(15) << "cpEnergyOnLayer\t"
                << std::setw(14) << "CPNhitsOnLayer\t"
                << std::setw(18) << "lcWithMaxEnergyInCP\t"
                << std::setw(15) << "maxEnergyLCinCP\t"
                << std::setw(20) << "CPEnergyFractionInLC" << "\n";
      LogDebug("HGCalValidator") << std::setw(8) << layerId << "\t"
                << std::setw(12) << cpId << "\t"
                << std::setw(15) << cP[cpId].energy() << "\t"
                << std::setw(15) << CPenergy << "\t"
                << std::setw(14) << CPNumberOfHits << "\t"
                << std::setw(18) << lcWithMaxEnergyInCP << "\t"
                << std::setw(15) << maxEnergyLCinCP << "\t"
                << std::setw(20) << CPEnergyFractionInLC << "\n";

      for(unsigned int i=0; i< CPNumberOfHits; ++i)
      {
        auto& cp_hitDetId = cPOnLayer[cpId][layerId].hits_and_fractions[i].first;
        auto& cpFraction = cPOnLayer[cpId][layerId].hits_and_fractions[i].second;


        bool hitWithNoLC = false;
        if(cpFraction ==0.f) continue; //hopefully this should never happen
        auto hit_find_in_LC = detIdToLayerClusterId_Map.find(cp_hitDetId);
        if(hit_find_in_LC == detIdToLayerClusterId_Map.end()) hitWithNoLC = true;
        auto itcheck= hitMap.find(cp_hitDetId);
        const HGCRecHit *hit = itcheck->second;
        float hitEnergyWeight = hit->energy()*hit->energy();
        float invCPEnergyWeight = 1.f/(CPenergy*CPenergy);
        for(auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore)
        {
          unsigned int layerClusterId = lcPair.first;
          float lcFraction = 0.f;

          if(!hitWithNoLC)
          {
            auto findHitIt = std::find(
                detIdToLayerClusterId_Map[cp_hitDetId].begin(),
                detIdToLayerClusterId_Map[cp_hitDetId].end(),
                HGVHistoProducerAlgo::detIdInfoInCluster{layerClusterId, 0.f}
                );
            if(findHitIt != detIdToLayerClusterId_Map[cp_hitDetId].end())
              lcFraction = findHitIt->fraction;
          }
          if (lcFraction == 0.) {
            lcFraction = -1.;
          }
          lcPair.second.second += (lcFraction - cpFraction)*(lcFraction - cpFraction)*hitEnergyWeight*invCPEnergyWeight;
          LogDebug("HGCalValidator") << "layerClusterId:\t" << layerClusterId << "\t"
                    << "lcfraction,cpfraction:\t" << lcFraction << ", " << cpFraction << "\t"
                    << "hitEnergyWeight:\t" << hitEnergyWeight << "\t"
                    << "currect score:\t" << lcPair.second.second << "\t"
                    << "invCPEnergyWeight:\t" << invCPEnergyWeight << "\n";
        }
      }



      if(cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore.empty())
        LogDebug("HGCalValidator") << "CP Id: \t" << cpId << "\tLC id:\t-1 " << "\t score \t-1" <<"\n";

      for(auto& lcPair : cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore)
      {
        LogDebug("HGCalValidator") << "CP Id: \t" << cpId << "\t LC id: \t"
                  << lcPair.first << "\t score \t"
                  << lcPair.second.second << "\t"
                  << "shared energy:\t" << lcPair.second.first << "\t"
                  << "shared energy fraction:\t" << (lcPair.second.first/CPenergy) << "\n";
        histograms.h_score_caloparticle2layercl_perlayer.at(layerId).fill(lcPair.second.second > 1. ? 1. : lcPair.second.second);
        histograms.h_sharedenergy_caloparticle2layercl_perlayer.at(layerId).fill(lcPair.second.first/CPenergy);
        histograms.h_energy_vs_score_caloparticle2layercl_perlayer.at(layerId).fill(lcPair.second.second  > 1. ? 1. : lcPair.second.second, lcPair.second.first/CPenergy);
      }
      auto assoc = std::count_if(
            std::begin(cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore),
            std::end(cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore),
            [](const auto &obj){return obj.second.second < ScoreCutCPtoLC_;});
      if (assoc) {
        histograms.h_num_caloparticle_eta_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().eta());
        histograms.h_num_caloparticle_phi_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().phi());
        if (assoc > 1) {
          histograms.h_numDup_caloparticle_eta_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().eta());
          histograms.h_numDup_caloparticle_phi_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().phi());
        }
        auto best = std::min_element(
            std::begin(cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore),
            std::end(cPOnLayer[cpId][layerId].layerClusterIdToEnergyAndScore),
              [](const auto &obj1, const auto &obj2){return obj1.second.second < obj2.second.second;});
        histograms.h_sharedenergy_caloparticle2layercl_vs_eta_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().eta(), best->second.first/CPenergy);
        histograms.h_sharedenergy_caloparticle2layercl_vs_phi_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().phi(), best->second.first/CPenergy);
      }
      histograms.h_denom_caloparticle_eta_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().eta());
      histograms.h_denom_caloparticle_phi_perlayer.at(layerId).fill(cP[cpId].g4Tracks()[0].momentum().phi());
    }
  }
}

void HGVHistoProducerAlgo::fill_generic_cluster_histos(const Histograms& histograms,
						       int count,
						       const reco::CaloClusterCollection &clusters, 
						       const Density &densities,
						       std::vector<CaloParticle> const & cP,
                                                       std::map<DetId, const HGCRecHit*> const & hitMap,
						       std::map<double, double> cummatbudg,
						       unsigned layers,
						       std::vector<int> thicknesses) const {
  
  //Each event to be treated as two events: an event in +ve endcap, 
  //plus another event in -ve endcap. In this spirit there will be 
  //a layer variable (layerid) that maps the layers in : 
  //-z: 0->51
  //+z: 52->103

  //To keep track of total num of layer clusters per layer
  //tnlcpl[layerid]
  std::vector<int> tnlcpl(1000, 0); //tnlcpl.clear(); tnlcpl.reserve(1000);
    
  //To keep track of the total num of clusters per thickness in plus and in minus endcaps
  std::map<std::string, int> tnlcpthplus; tnlcpthplus.clear();
  std::map<std::string, int> tnlcpthminus; tnlcpthminus.clear();
  //At the beginning of the event all layers should be initialized to zero total clusters per thickness
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) { 
    tnlcpthplus.insert( std::pair<std::string, int>(std::to_string(*it), 0) ); 
    tnlcpthminus.insert( std::pair<std::string, int>(std::to_string(*it), 0) ); 
  }
  //To keep track of the total num of clusters with mixed thickness hits per event
  tnlcpthplus.insert( std::pair<std::string, int>( "mixed", 0) ); 
  tnlcpthminus.insert( std::pair<std::string, int>( "mixed", 0) ); 

  layerClusters_to_CaloParticles(histograms, clusters, cP, hitMap, layers);

  //To find out the total amount of energy clustered per layer
  //Initialize with zeros because I see clear gives weird numbers. 
  std::vector<double> tecpl(1000, 0.0); //tecpl.clear(); tecpl.reserve(1000);
  //for the longitudinal depth barycenter
  std::vector<double> ldbar(1000, 0.0); //ldbar.clear(); ldbar.reserve(1000);
  
  //We need to compare with the total amount of energy coming from caloparticles
  double caloparteneplus = 0.;
  double caloparteneminus = 0.;
  for (auto const caloParticle : cP) {
    if (caloParticle.eta() >= 0. ) {caloparteneplus = caloparteneplus + caloParticle.energy();}
    if (caloParticle.eta() < 0. )  {caloparteneminus = caloparteneminus + caloParticle.energy();}
  }

  //loop through clusters of the event
  for (unsigned int layerclusterIndex = 0; layerclusterIndex < clusters.size(); layerclusterIndex++) {

    const std::vector<std::pair<DetId, float> > hits_and_fractions = clusters[layerclusterIndex].hitsAndFractions();

    const DetId seedid = clusters[layerclusterIndex].seed();
    const double seedx = recHitTools_->getPosition(seedid).x();
    const double seedy = recHitTools_->getPosition(seedid).y();
    DetId maxid = findmaxhit( clusters[layerclusterIndex], hitMap );

    // const DetId maxid = clusters[layerclusterIndex].max();
    double maxx = recHitTools_->getPosition(maxid).x();
    double maxy = recHitTools_->getPosition(maxid).y();

    //Auxillary variables to count the number of different kind of hits in each cluster
    int nthhits120p = 0; int nthhits200p = 0;int nthhits300p = 0;int nthhitsscintp = 0;
    int nthhits120m = 0; int nthhits200m = 0;int nthhits300m = 0;int nthhitsscintm = 0;
    //For the hits thickness of the layer cluster. 
    double thickness = 0.;
    //The layer the cluster belongs to. As mentioned in the mapping above, it takes into account -z and +z. 
    int layerid = 0;
    //We will need another layer variable for the longitudinal material budget file reading. 
    //In this case we need no distinction between -z and +z. 
    int lay = 0;
    //We will need here to save the combination thick_lay 
    std::string istr = "";
    //boolean to check for the layer that the cluster belong to. Maybe later will check all the layer hits.
    bool cluslay = true;
    //zside that the current cluster belongs to. 
    int zside = 0;

    for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin(); it_haf != hits_and_fractions.end(); ++it_haf) {
      const DetId rh_detid = it_haf->first;
      //The layer that the current hit belongs to
      layerid = recHitTools_->getLayerWithOffset(rh_detid) + layers * ((recHitTools_->zside(rh_detid) + 1) >> 1) - 1;
      lay = recHitTools_->getLayerWithOffset(rh_detid);
      zside = recHitTools_->zside(rh_detid);
      if (rh_detid.det() == DetId::Forward || rh_detid.det() == DetId::HGCalEE || rh_detid.det() == DetId::HGCalHSi){
	thickness = recHitTools_->getSiThickness(rh_detid);
      } else if (rh_detid.det() == DetId::HGCalHSc){
	thickness = -1;
      } else {
      	LogDebug("HGCalValidator") << "These are HGCal layer clusters, you shouldn't be here !!! " << layerid  << "\n";
      	continue;
      }

      //Count here only once the layer cluster and save the combination thick_layerid
      std::string curistr = std::to_string( (int) thickness );
      std::string lay_string = std::to_string(layerid);
      while(lay_string.size() < 2)
        lay_string.insert(0, "0");
      curistr += "_" + lay_string;
      if (cluslay){ tnlcpl[layerid]++; istr = curistr; cluslay = false; }

      if ( (thickness == 120.) && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhits120p++;}
      else if ( (thickness == 120.) && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhits120m++;}
      else if ( (thickness == 200.) && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhits200p++;}
      else if ( (thickness == 200.) && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhits200m++;}
      else if ( (thickness == 300.) && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhits300p++;}
      else if ( (thickness == 300.) && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhits300m++;}
      else if ( (thickness == -1)   && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhitsscintp++;}
      else if ( (thickness == -1)   && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhitsscintm++;}
      else { //assert(0);
	LogDebug("HGCalValidator") << " You are running a geometry that contains thicknesses different than the normal ones. " << "\n";
      }

      std::map<DetId,const HGCRecHit *>::const_iterator itcheck= hitMap.find(rh_detid);
      if (itcheck == hitMap.end()) {
	LogDebug("HGCalValidator") << " You shouldn't be here - Unable to find a hit " << rh_detid.rawId() << " " << rh_detid.det() << " " << HGCalDetId(rh_detid) << "\n";
	continue;
      }

      const HGCRecHit *hit = itcheck->second;

      //Here for the per cell plots
      //----
      const double hit_x = recHitTools_->getPosition(rh_detid).x();
      const double hit_y = recHitTools_->getPosition(rh_detid).y();
      double distancetoseed = distance(seedx, seedy, hit_x, hit_y);
      double distancetomax = distance(maxx, maxy, hit_x, hit_y);
      if ( distancetoseed != 0. && histograms.h_distancetoseedcell_perthickperlayer.count(curistr)){ 
      	histograms.h_distancetoseedcell_perthickperlayer.at(curistr).fill( distancetoseed  );    
      } 
      //----
      if ( distancetoseed != 0. && histograms.h_distancetoseedcell_perthickperlayer_eneweighted.count(curistr)){
      	histograms.h_distancetoseedcell_perthickperlayer_eneweighted.at(curistr).fill( distancetoseed , hit->energy() );    }
      //----
      if ( distancetomax != 0. && histograms.h_distancetomaxcell_perthickperlayer.count(curistr)){
      	histograms.h_distancetomaxcell_perthickperlayer.at(curistr).fill( distancetomax  );    }
      //----
      if ( distancetomax != 0. && histograms.h_distancetomaxcell_perthickperlayer_eneweighted.count(curistr)){ 
      	histograms.h_distancetomaxcell_perthickperlayer_eneweighted.at(curistr).fill( distancetomax , hit->energy() );    } 

      //Let's check the density
      std::map< DetId, float >::const_iterator dit = densities.find( rh_detid );
      if ( dit == densities.end() ){
      	LogDebug("HGCalValidator") << " You shouldn't be here - Unable to find a density " << rh_detid.rawId() << " " << rh_detid.det() << " " << HGCalDetId(rh_detid) << "\n";
      	continue;
      }

      if ( histograms.h_cellsenedens_perthick.count( (int) thickness ) ){
	histograms.h_cellsenedens_perthick.at( (int) thickness ).fill( dit->second );
      }
	
    } // end of loop through hits and fractions

    //Check for simultaneously having hits of different kind. Checking at least two combinations is sufficient.
    if ( (nthhits120p != 0 && nthhits200p != 0  ) || (nthhits120p != 0 && nthhits300p != 0  ) || (nthhits120p != 0 && nthhitsscintp != 0  ) || 
	 (nthhits200p != 0 && nthhits300p != 0  ) || (nthhits200p != 0 && nthhitsscintp != 0  ) || (nthhits300p != 0 && nthhitsscintp != 0  ) ){
      tnlcpthplus["mixed"]++;
    } else if ( (nthhits120p != 0 ||  nthhits200p != 0 || nthhits300p != 0 || nthhitsscintp != 0) )  {
      //This is a cluster with hits of one kind
      tnlcpthplus[std::to_string((int) thickness)]++;
    }
    if ( (nthhits120m != 0 && nthhits200m != 0  ) || (nthhits120m != 0 && nthhits300m != 0  ) || (nthhits120m != 0 && nthhitsscintm != 0  ) || 
	 (nthhits200m != 0 && nthhits300m != 0  ) || (nthhits200m != 0 && nthhitsscintm != 0  ) || (nthhits300m != 0 && nthhitsscintm != 0  ) ){
      tnlcpthminus["mixed"]++;
    } else if ( (nthhits120m != 0 ||  nthhits200m != 0 || nthhits300m != 0 || nthhitsscintm != 0) )  {
      //This is a cluster with hits of one kind
      tnlcpthminus[std::to_string((int) thickness)]++;
    }

    //To find the thickness with the biggest amount of cells
    std::vector<int> bigamoth; bigamoth.clear();
    if (zside > 0 ){
      bigamoth.push_back(nthhits120p);bigamoth.push_back(nthhits200p);bigamoth.push_back(nthhits300p);bigamoth.push_back(nthhitsscintp);
    }
    if (zside < 0 ){
      bigamoth.push_back(nthhits120m);bigamoth.push_back(nthhits200m);bigamoth.push_back(nthhits300m);bigamoth.push_back(nthhitsscintm);
    }
    auto bgth = std::max_element(bigamoth.begin(),bigamoth.end());
    istr = std::to_string(thicknesses[ std::distance(bigamoth.begin(), bgth) ]);
    std::string lay_string = std::to_string(layerid);
    while(lay_string.size() < 2)
      lay_string.insert(0, "0");
    istr += "_" + lay_string;

    //Here for the per cluster plots that need the thickness_layer info
    if (histograms.h_cellsnum_perthickperlayer.count(istr)){ histograms.h_cellsnum_perthickperlayer.at(istr).fill( hits_and_fractions.size() ); }
    
    //Now, with the distance between seed and max cell. 
    double distancebetseedandmax = distance(seedx, seedy, maxx, maxy);
    //The thickness_layer combination in this case will use the thickness of the seed as a convention. 
    std::string seedstr = std::to_string( (int) recHitTools_->getSiThickness(seedid) )+ "_" + std::to_string(layerid);
    seedstr += "_" + lay_string;
    if (histograms.h_distancebetseedandmaxcell_perthickperlayer.count(seedstr)){ 
      histograms.h_distancebetseedandmaxcell_perthickperlayer.at(seedstr).fill( distancebetseedandmax ); 
    }
    if (histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer.count(seedstr)){ 
      histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer.at(seedstr).fill( distancebetseedandmax , clusters[layerclusterIndex].energy() ); 
    }

    //Energy clustered per layer
    tecpl[layerid] = tecpl[layerid] + clusters[layerclusterIndex].energy();
    ldbar[layerid] = ldbar[layerid] + clusters[layerclusterIndex].energy() * cummatbudg[(double) lay];

  }//end of loop through clusters of the event

  //After the end of the event we can now fill with the results. 
  //First a couple of variables to keep the sum of the energy of all clusters
  double sumeneallcluspl = 0.; double sumeneallclusmi = 0.;
  //And the longitudinal variable
  double sumldbarpl = 0.; double sumldbarmi = 0.;
  //Per layer : Loop 0->103
  for (unsigned ilayer = 0; ilayer < layers*2; ++ilayer) {
    if (histograms.h_clusternum_perlayer.count(ilayer)){ 
      histograms.h_clusternum_perlayer.at(ilayer).fill( tnlcpl[ilayer] ); 
    }
    // Two times one for plus and one for minus
    //First with the -z endcap
    if (ilayer < layers){
      if (histograms.h_energyclustered_perlayer.count(ilayer)){ 
	if ( caloparteneminus != 0.) {
	  histograms.h_energyclustered_perlayer.at(ilayer).fill( 100. * tecpl[ilayer]/caloparteneminus ); 
	}
      }
      //Keep here the total energy for the event in -z
      sumeneallclusmi = sumeneallclusmi + tecpl[ilayer];
      //And for the longitudinal variable
      sumldbarmi = sumldbarmi + ldbar[ilayer];
    } else { //Then for the +z
      if (histograms.h_energyclustered_perlayer.count(ilayer)){ 
	if ( caloparteneplus != 0.) {
	  histograms.h_energyclustered_perlayer.at(ilayer).fill( 100. * tecpl[ilayer]/caloparteneplus ); 
	}
      }
      //Keep here the total energy for the event in -z
      sumeneallcluspl = sumeneallcluspl + tecpl[ilayer];
      //And for the longitudinal variable
      sumldbarpl = sumldbarpl + ldbar[ilayer];
    } //end of +z loop 

  }//end of loop over layers 
    
  //Per thickness
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) { 
    if ( histograms.h_clusternum_perthick.count(*it) ){ 
      histograms.h_clusternum_perthick.at(*it).fill( tnlcpthplus[std::to_string(*it)] ); 
      histograms.h_clusternum_perthick.at(*it).fill( tnlcpthminus[std::to_string(*it)] ); 
    } 
  }
  //Mixed thickness clusters
  histograms.h_mixedhitscluster_zplus[count].fill( tnlcpthplus["mixed"]  );
  histograms.h_mixedhitscluster_zminus[count].fill( tnlcpthminus["mixed"]  );

  //Total energy clustered from all layer clusters (fraction)
  if ( caloparteneplus != 0.) {histograms.h_energyclustered_zplus[count].fill( 100. * sumeneallcluspl /caloparteneplus ); }
  if ( caloparteneminus != 0.) {histograms.h_energyclustered_zminus[count].fill( 100. * sumeneallclusmi /caloparteneminus ); }

  //For the longitudinal depth barycenter
  histograms.h_longdepthbarycentre_zplus[count].fill( sumldbarpl / sumeneallcluspl ); 
  histograms.h_longdepthbarycentre_zminus[count].fill( sumldbarmi / sumeneallclusmi ); 

}

double HGVHistoProducerAlgo::distance2(const double x1, const double y1, const double x2, const double y2) const {   //distance squared
  const double dx = x1 - x2;
  const double dy = y1 - y2;
  return (dx*dx + dy*dy);
}   //distance squaredq
double HGVHistoProducerAlgo::distance(const double x1, const double y1, const double x2, const double y2) const{ //2-d distance on the layer (x-y)
  return std::sqrt(distance2(x1,y1,x2,y2) );
}

void HGVHistoProducerAlgo::setRecHitTools(std::shared_ptr<hgcal::RecHitTools> recHitTools) {
  recHitTools_ = recHitTools;
}

DetId HGVHistoProducerAlgo::findmaxhit(const reco::CaloCluster & cluster,
    std::map<DetId, const HGCRecHit*> const & hitMap) const {

  DetId themaxid;
  const std::vector<std::pair<DetId, float> >& hits_and_fractions = cluster.hitsAndFractions();

  double maxene = 0.;
  for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin(); it_haf != hits_and_fractions.end(); ++it_haf) {
    DetId rh_detid = it_haf->first;

    std::map<DetId,const HGCRecHit *>::const_iterator itcheck= hitMap.find(rh_detid);
    const HGCRecHit *hit = itcheck->second;

    if ( maxene < hit->energy() ){
      maxene = hit->energy();
      themaxid = rh_detid;
    }

  }

  return themaxid;
}


double HGVHistoProducerAlgo::getEta(double eta) const {
  if (useFabsEta_) return fabs(eta);
  else return eta;
}

