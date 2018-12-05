#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"
#include <TF1.h>

using namespace std;

HGVHistoProducerAlgo::HGVHistoProducerAlgo(const edm::ParameterSet& pset) {
  //parameters for eta plots
  minEta  = pset.getParameter<double>("minEta");
  maxEta  = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for z positionof vertex plots 
  minZpos  = pset.getParameter<double>("minZpos");
  maxZpos  = pset.getParameter<double>("maxZpos");
  nintZpos = pset.getParameter<int>("nintZpos");

  //Parameters for the total number of layer clusters per layer
  minTotNClsperlay  = pset.getParameter<double>("minTotNClsperlay");
  maxTotNClsperlay  = pset.getParameter<double>("maxTotNClsperlay");
  nintTotNClsperlay = pset.getParameter<int>("nintTotNClsperlay");

  //Parameters for the total number of layer clusters per thickness
  minTotNClsperthick  = pset.getParameter<double>("minTotNClsperthick");
  maxTotNClsperthick  = pset.getParameter<double>("maxTotNClsperthick");
  nintTotNClsperthick = pset.getParameter<int>("nintTotNClsperthick");

  //Parameters for the total number of cells per per thickness per layer
  minTotNcellsperthickperlayer  = pset.getParameter<double>("minTotNcellsperthickperlayer");
  maxTotNcellsperthickperlayer  = pset.getParameter<double>("maxTotNcellsperthickperlayer");
  nintTotNcellsperthickperlayer = pset.getParameter<int>("nintTotNcellsperthickperlayer");

  //Parameters for the distance of cluster cells to seed cell per thickness per layer
  minDisToSeedperthickperlayer  = pset.getParameter<double>("minDisToSeedperthickperlayer");
  maxDisToSeedperthickperlayer  = pset.getParameter<double>("maxDisToSeedperthickperlayer");
  nintDisToSeedperthickperlayer = pset.getParameter<int>("nintDisToSeedperthickperlayer");

  //Parameters for the energy weighted distance of cluster cells to seed cell per thickness per layer
  minDisToSeedperthickperlayerenewei  = pset.getParameter<double>("minDisToSeedperthickperlayerenewei");
  maxDisToSeedperthickperlayerenewei  = pset.getParameter<double>("maxDisToSeedperthickperlayerenewei");
  nintDisToSeedperthickperlayerenewei = pset.getParameter<int>("nintDisToSeedperthickperlayerenewei");

  //Parameters for the distance of cluster cells to max cell per thickness per layer
  minDisToMaxperthickperlayer  = pset.getParameter<double>("minDisToMaxperthickperlayer");
  maxDisToMaxperthickperlayer  = pset.getParameter<double>("maxDisToMaxperthickperlayer");
  nintDisToMaxperthickperlayer = pset.getParameter<int>("nintDisToMaxperthickperlayer");

  //Parameters for the energy weighted distance of cluster cells to max cell per thickness per layer
  minDisToMaxperthickperlayerenewei  = pset.getParameter<double>("minDisToMaxperthickperlayerenewei");
  maxDisToMaxperthickperlayerenewei  = pset.getParameter<double>("maxDisToMaxperthickperlayerenewei");
  nintDisToMaxperthickperlayerenewei = pset.getParameter<int>("nintDisToMaxperthickperlayerenewei");

  //Parameters for the energy density of cluster cells per thickness 
  minCellsEneDensperthick  = pset.getParameter<double>("minCellsEneDensperthick");
  maxCellsEneDensperthick  = pset.getParameter<double>("maxCellsEneDensperthick");
  nintCellsEneDensperthick = pset.getParameter<int>("nintCellsEneDensperthick");




}

HGVHistoProducerAlgo::~HGVHistoProducerAlgo() {}

void HGVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid) {

  histograms.h_caloparticle_eta[pdgid] = ibook.book1D("num_caloparticle_eta","N of caloparticle vs eta",nintEta,minEta,maxEta);
  histograms.h_caloparticle_eta_Zorigin[pdgid] = ibook.book2D("Eta vs Zorigin", "Eta vs Zorigin", nintEta, minEta, maxEta, nintZpos, minZpos, maxZpos);
  
}


void HGVHistoProducerAlgo::bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms, unsigned layers, std::vector<int> thicknesses) {

  //---------------------------------------------------------------------------------------------------------------------------
  histograms.h_cluster_eta.push_back( ibook.book1D("num_reco_cluster_eta","N of reco clusters vs eta",nintEta,minEta,maxEta) );

  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 1; ilayer <= layers; ++ilayer) {
    auto istr = std::to_string(ilayer);
    histograms.h_clusternum_perlayer[ilayer] = ibook.book1D("totclusternum_layer_"+istr,"total number of layer clusters for layer "+istr,nintTotNClsperlay,minTotNClsperlay,maxTotNClsperlay);
  }

  //---------------------------------------------------------------------------------------------------------------------------
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    auto istr = std::to_string(*it);
    histograms.h_clusternum_perthick[(*it)] = ibook.book1D("totclusternum_thick_"+istr,"total number of layer clusters for thickness "+istr,nintTotNClsperthick,minTotNClsperthick,maxTotNClsperthick);
    //---
    histograms.h_cellsenedens_perthick[(*it)] = ibook.book1D("cellsenedens_thick_"+istr,"energy density of cluster cells for thickness "+istr,nintCellsEneDensperthick,minCellsEneDensperthick,maxCellsEneDensperthick);
  }

  //---------------------------------------------------------------------------------------------------------------------------
  //Not all combination exists but we should keep them all for cross checking reason. 
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) {
    for (unsigned ilayer = 1; ilayer <= layers; ++ilayer) {
      auto istr1 = std::to_string(*it);
      auto istr2 = std::to_string(ilayer);
      auto istr = istr1 + "_" + istr2;
      //---
      histograms.h_cellsnum_perthickperlayer[istr] = ibook.book1D("cellsnum_perthick_perlayer_"+istr,"total number of cells for layer "+ istr1+" for thickness "+istr2,nintTotNcellsperthickperlayer,minTotNcellsperthickperlayer,maxTotNcellsperthickperlayer);
      //---
       histograms.h_distancetoseedcell_perthickperlayer[istr] = ibook.book1D("distancetoseedcell_perthickperlayer_"+istr,"distance of cluster cells to seed cell for layer "+ istr1+" for thickness "+istr2,nintDisToSeedperthickperlayer,minDisToSeedperthickperlayer,maxDisToSeedperthickperlayer);
      //---
       histograms.h_distancetoseedcell_perthickperlayer_eneweighted[istr] = ibook.book1D("distancetoseedcell_perthickperlayer_eneweighted_"+istr,"energy weighted distance of cluster cells to seed cell for layer "+ istr1+" for thickness "+istr2,nintDisToSeedperthickperlayerenewei,minDisToSeedperthickperlayerenewei,maxDisToSeedperthickperlayerenewei);
      //---
       histograms.h_distancetomaxcell_perthickperlayer[istr] = ibook.book1D("distancetomaxcell_perthickperlayer_"+istr,"distance of cluster cells to max cell for layer "+ istr1+" for thickness "+istr2,nintDisToMaxperthickperlayer,minDisToMaxperthickperlayer,maxDisToMaxperthickperlayer);
      //---
       histograms.h_distancetomaxcell_perthickperlayer_eneweighted[istr] = ibook.book1D("distancetomaxcell_perthickperlayer_eneweighted_"+istr,"energy weighted distance of cluster cells to max cell for layer "+ istr1+" for thickness "+istr2,nintDisToMaxperthickperlayerenewei,minDisToMaxperthickperlayerenewei,maxDisToMaxperthickperlayerenewei);
    }
  }
  //---------------------------------------------------------------------------------------------------------------------------


  

}


void HGVHistoProducerAlgo::fill_caloparticle_histos(const Histograms& histograms,
						    int pdgid,
						    const CaloParticle & caloparticle,
						    std::vector<SimVertex> const & simVertices) const {

  const auto eta = getEta(caloparticle.eta());
  if (histograms.h_caloparticle_eta.count(pdgid)){ histograms.h_caloparticle_eta.at(pdgid).fill(eta); }
  if (histograms.h_caloparticle_eta_Zorigin.count(pdgid)){ histograms.h_caloparticle_eta_Zorigin.at(pdgid).fill( simVertices.at(caloparticle.g4Tracks()[0].vertIndex()).position().z(), eta ); }

}

void HGVHistoProducerAlgo::fill_cluster_histos(const Histograms& histograms,
					       int count,
					       const reco::CaloCluster & cluster) const {

  const auto eta = getEta(cluster.eta());
  histograms.h_cluster_eta[count].fill(eta);
}

void HGVHistoProducerAlgo::fill_generic_cluster_histos(const Histograms& histograms,
						       int count,
						       const reco::CaloClusterCollection &clusters, 
						       const edm::EventSetup& setup,
						       unsigned layers, 
						       std::vector<int> thicknesses) const {
  
  
  recHitTools_->getEventSetup(setup);
  //To keep track of total num of layer clusters per layer
  //tnlcpl[lay]
  std::vector<int> tnlcpl; tnlcpl.clear(); tnlcpl.reserve(100);
  //At the beginning of the event all layers should be initialized to zero total clusters
  for (unsigned ilayer = 1; ilayer <= layers+1; ++ilayer) {
    //Be careful here we should push back to layers plus 1 since layers starts from 1.  
    tnlcpl.push_back(0);
  }
    
  //To keep track of the total num of clusters per thickness
  std::map<std::string, int> tnlcpth; tnlcpth.clear();
  //At the beginning of the event all layers should be initialized to zero total clusters per thickness
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) { 
    tnlcpth.insert( std::pair<std::string, int>(std::to_string(*it), 0) ); 
  }

  //loop through clusters of the event
  for (unsigned int layerclusterIndex = 0; layerclusterIndex < clusters.size(); layerclusterIndex++) {
    
    const std::vector<std::pair<DetId, float> > hits_and_fractions = clusters[layerclusterIndex].hitsAndFractions();

    // const DetId seedid = clusters[layerclusterIndex].seed();
    // const double seedx = recHitTools_->getPosition(seedid).x();
    // const double seedy = recHitTools_->getPosition(seedid).y();
    // const DetId maxid = clusters[layerclusterIndex].max();
    // double maxx = recHitTools_->getPosition(maxid).x();
    // double maxy = recHitTools_->getPosition(maxid).y();


   //We will need here to save the combination thick_lay for the first hit of the hits of the cluster
    std::string istr = "";
    //boolean because we need only once to check for the layer that the cluster belong to. No need to go though all the layer hits.  
    bool cluslay = true;
    //For each cluster we check one hit and depending on the thickness of that hit we marked it as 
    //"cluster of that thickness". However, other hit thicknesses may appear but for now we don't take this into account. 
    bool clusthick120 = true;
    bool clusthick200 = true;
    bool clusthick300 = true;
    bool clusthickscint = true;
    for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin(); it_haf != hits_and_fractions.end(); ++it_haf) {
      const DetId rh_detid = it_haf->first;
      //The layer that the current hit belongs to
      //I do not know why but it returns also 101 and 104 as values
      int lay = recHitTools_->getLayerWithOffset(rh_detid);
      double thickness = (rh_detid.det() == DetId::Forward || rh_detid.det() == DetId::HGCalEE || rh_detid.det() == DetId::HGCalHSi) ? recHitTools_->getSiThickness(rh_detid) : -1;
      //Count here only once the layer cluster and save the combination thick_lay
      std::string curistr = std::to_string( (int) thickness )  + std::to_string(lay);
      if (cluslay){ tnlcpl[lay]++; istr = curistr; cluslay = false; }
      if (thickness == 120. && clusthick120  ) {tnlcpth[std::to_string((int) thickness)]++;clusthick120 = false;}
      if (thickness == 200. && clusthick200  ) {tnlcpth[std::to_string((int) thickness)]++;clusthick200 = false;}
      if (thickness == 300. && clusthick300  ) {tnlcpth[std::to_string((int) thickness)]++;clusthick300 = false;}
      if (thickness == -1 && clusthickscint  ) {tnlcpth[std::to_string((int) thickness)]++;clusthickscint = false;}

      //I do not break the loop, it must be needed later for the per cell plots
      // break;

      //Here for the per cell plots
      //----
      // const double hit_x = recHitTools_->getPosition(rh_detid).x();
      // const double hit_y = recHitTools_->getPosition(rh_detid).y();
      // double distancetoseed = distance(seedx, seedy, hit_x, hit_y);
      // if (histograms.h_distancetoseedcell_perthickperlayer.count(curistr)){ 
      // 	histograms.h_distancetoseedcell_perthickperlayer.at(curistr).fill( distancetoseed  );    
      // } 
      //----
      // if (histograms.h_distancetoseedcell_perthickperlayer_eneweighted.count(curistr)){ 
      // 	histograms.h_distancetoseedcell_perthickperlayer_eneweighted.at(curistr).fill( distance(seedx,seedy,recHitTools_->getPosition(rh_detid).x(),recHitTools_->getPosition(rh_detid).y()) , ENERGYOFRECHITHERE );    } 
      //----
      // if (histograms.h_distancetomaxcell_perthickperlayer.count(curistr)){ 
      // 	histograms.h_distancetomaxcell_perthickperlayer.at(curistr).fill( distance(maxx,maxy,recHitTools_->getPosition(rh_detid).x(),recHitTools_->getPosition(rh_detid).y())  );    } 
      //----
      // if (histograms.h_distancetomaxcell_perthickperlayer_eneweighted.count(curistr)){ 
      // 	histograms.h_distancetomaxcell_perthickperlayer_eneweighted.at(curistr).fill( distance(maxx,maxy,recHitTools_->getPosition(rh_detid).x(),recHitTools_->getPosition(rh_detid).y()) , ENERGYOFRECHITHERE );    } 
      //----
      
      
      
    } // end of loop through hits and fractions

    //Here for the per cluster plots that need the thickness_layer info
    if (histograms.h_cellsnum_perthickperlayer.count(istr)){ histograms.h_cellsnum_perthickperlayer.at(istr).fill( hits_and_fractions.size() ); }
      

  }//end of loop through clusters of the event

  //After the end of the event we can now fill with the results. 
  //Per layer
  for (unsigned ilayer = 1; ilayer <= layers; ++ilayer) {
    if (histograms.h_clusternum_perlayer.count(ilayer)){ histograms.h_clusternum_perlayer.at(ilayer).fill( tnlcpl[ilayer] ); }
  }
  //Per thickness
  for(std::vector<int>::iterator it = thicknesses.begin(); it != thicknesses.end(); ++it) { 
    if ( histograms.h_clusternum_perthick.count(*it) ){ 
      histograms.h_clusternum_perthick.at(*it).fill( tnlcpth[std::to_string(*it)]); 
    } 
  }
  



}

double HGVHistoProducerAlgo::distance2(const double x1, const double y1, const double x2, const double y2) const {   //distance squared
  const double dx = x1 - x2;
  const double dy = y1 - y2;
  return (dx*dx + dy*dy);
}   //distance squaredq
double HGVHistoProducerAlgo::distance(const double x1, const double y1, const double x2, const double y2) const{ //2-d distance on the layer (x-y)
  return std::sqrt(distance2(x1,y1,x2,y2) );
}


void HGVHistoProducerAlgo::setRecHitTools(hgcal::RecHitTools * recHitTools ) {
  recHitTools_ = recHitTools;
}

double HGVHistoProducerAlgo::getEta(double eta) const {
  if (useFabsEta) return fabs(eta);
  else return eta;
}

