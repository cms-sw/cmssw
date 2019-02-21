#include "Validation/HGCalValidation/interface/HGVHistoProducerAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"
#include "TLatex.h"
#include <TF1.h>
#include <numeric> 

using namespace std;

HGVHistoProducerAlgo::HGVHistoProducerAlgo(const edm::ParameterSet& pset) {
  hitMap_ = new std::map<DetId, const HGCRecHit *>();

  //parameters for calo particles eta 
  minEta  = pset.getParameter<double>("minEta");
  maxEta  = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for calo particles energy 
  minCaloEne  = pset.getParameter<double>("minCaloEne");
  maxCaloEne  = pset.getParameter<double>("maxCaloEne");
  nintCaloEne = pset.getParameter<int>("nintCaloEne");

  //parameters for calo particles pt 
  minCaloPt  = pset.getParameter<double>("minCaloPt");
  maxCaloPt  = pset.getParameter<double>("maxCaloPt");
  nintCaloPt = pset.getParameter<int>("nintCaloPt");

  //parameters for calo particles phi
  minCaloPhi  = pset.getParameter<double>("minCaloPhi");
  maxCaloPhi  = pset.getParameter<double>("maxCaloPhi");
  nintCaloPhi = pset.getParameter<int>("nintCaloPhi");

  //parameters for counting mixed hits cluster
  minMixedHitsCluster  = pset.getParameter<double>("minMixedHitsCluster");
  maxMixedHitsCluster  = pset.getParameter<double>("maxMixedHitsCluster");
  nintMixedHitsCluster = pset.getParameter<int>("nintMixedHitsCluster");

  //parameters for the total amount of energy clustered by all layer clusters (fraction over caloparticles)
  minEneCl  = pset.getParameter<double>("minEneCl");
  maxEneCl  = pset.getParameter<double>("maxEneCl");
  nintEneCl = pset.getParameter<int>("nintEneCl");

  //parameters for the longitudinal depth barycenter. 
  minLongDepBary  = pset.getParameter<double>("minLongDepBary");
  maxLongDepBary  = pset.getParameter<double>("maxLongDepBary");
  nintLongDepBary = pset.getParameter<int>("nintLongDepBary");

  //parameters for z positionof vertex plots 
  minZpos  = pset.getParameter<double>("minZpos");
  maxZpos  = pset.getParameter<double>("maxZpos");
  nintZpos = pset.getParameter<int>("nintZpos");

  //Parameters for the total number of layer clusters per layer
  minTotNClsperlay  = pset.getParameter<double>("minTotNClsperlay");
  maxTotNClsperlay  = pset.getParameter<double>("maxTotNClsperlay");
  nintTotNClsperlay = pset.getParameter<int>("nintTotNClsperlay");

  //Parameters for the energy clustered by layer clusters per layer (fraction over caloparticles)
  minEneClperlay  = pset.getParameter<double>("minEneClperlay");
  maxEneClperlay  = pset.getParameter<double>("maxEneClperlay");
  nintEneClperlay = pset.getParameter<int>("nintEneClperlay");

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

  //Parameters for the distance of seed cell to max cell per thickness per layer
  minDisSeedToMaxperthickperlayer  = pset.getParameter<double>("minDisSeedToMaxperthickperlayer");
  maxDisSeedToMaxperthickperlayer  = pset.getParameter<double>("maxDisSeedToMaxperthickperlayer");
  nintDisSeedToMaxperthickperlayer = pset.getParameter<int>("nintDisSeedToMaxperthickperlayer");

  //Parameters for the energy of a cluster per thickness per layer
  minClEneperthickperlayer  = pset.getParameter<double>("minClEneperthickperlayer");
  maxClEneperthickperlayer = pset.getParameter<double>("maxClEneperthickperlayer");
  nintClEneperthickperlayer = pset.getParameter<int>("nintClEneperthickperlayer");

  //Parameters for the energy density of cluster cells per thickness 
  minCellsEneDensperthick  = pset.getParameter<double>("minCellsEneDensperthick");
  maxCellsEneDensperthick  = pset.getParameter<double>("maxCellsEneDensperthick");
  nintCellsEneDensperthick = pset.getParameter<int>("nintCellsEneDensperthick");


}

HGVHistoProducerAlgo::~HGVHistoProducerAlgo() {}

void HGVHistoProducerAlgo::bookCaloParticleHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms,int pdgid) {

  histograms.h_caloparticle_eta[pdgid] = ibook.book1D("num_caloparticle_eta","N of caloparticle vs eta",nintEta,minEta,maxEta);
  histograms.h_caloparticle_eta_Zorigin[pdgid] = ibook.book2D("Eta vs Zorigin", "Eta vs Zorigin", nintEta, minEta, maxEta, nintZpos, minZpos, maxZpos);

  histograms.h_caloparticle_energy[pdgid] = ibook.book1D("caloparticle_energy", "Energy of caloparticle", nintCaloEne,minCaloEne,maxCaloEne);
  histograms.h_caloparticle_pt[pdgid] = ibook.book1D("caloparticle_pt", "Pt of caloparticle", nintCaloPt,minCaloPt,maxCaloPt);
  histograms.h_caloparticle_phi[pdgid] = ibook.book1D("caloparticle_phi", "Phi of caloparticle", nintCaloPhi,minCaloPhi,maxCaloPhi);

  
}


void HGVHistoProducerAlgo::bookClusterHistos(DQMStore::ConcurrentBooker& ibook, Histograms& histograms, unsigned layers, std::vector<int> thicknesses) {

  //---------------------------------------------------------------------------------------------------------------------------
  histograms.h_cluster_eta.push_back( ibook.book1D("num_reco_cluster_eta","N of reco clusters vs eta",nintEta,minEta,maxEta) );
  
  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_mixedhitscluster_zminus.push_back( ibook.book1D("mixedhitscluster_zminus","N of reco clusters that contain hits of more than one kind in z-",nintMixedHitsCluster,minMixedHitsCluster,maxMixedHitsCluster) );
  //z+
  histograms.h_mixedhitscluster_zplus.push_back( ibook.book1D("mixedhitscluster_zplus","N of reco clusters that contain hits of more than one kind in z+",nintMixedHitsCluster,minMixedHitsCluster,maxMixedHitsCluster) );
  
  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_energyclustered_zminus.push_back( ibook.book1D("energyclustered_zminus","percent of total energy clustered by all layer clusters over caloparticles energy in z-",nintEneCl,minEneCl,maxEneCl) );
  //z+
  histograms.h_energyclustered_zplus.push_back( ibook.book1D("energyclustered_zplus","percent of total energy clustered by all layer clusters over caloparticles energy in z+",nintEneCl,minEneCl,maxEneCl) );
  
  //---------------------------------------------------------------------------------------------------------------------------
  //z-
  histograms.h_longdepthbarycentre_zminus.push_back( ibook.book1D("longdepthbarycentre_zminus","The longitudinal depth barycentre in z-",nintLongDepBary,minLongDepBary,maxLongDepBary) );
  //z+
  histograms.h_longdepthbarycentre_zplus.push_back( ibook.book1D("longdepthbarycentre_zplus","The longitudinal depth barycentre in z+",nintLongDepBary,minLongDepBary,maxLongDepBary) );


  //---------------------------------------------------------------------------------------------------------------------------
  for (unsigned ilayer = 0; ilayer < 2*layers; ++ilayer) {
    auto istr1 = std::to_string(ilayer);
    //We will make a mapping to the regural layer naming plus z- or z+ for convenience
    std::string istr2 = "";
     //First with the -z endcap
    if (ilayer < layers){
      istr2 = std::to_string(ilayer + 1) + " in z-";
    }else { //Then for the +z
      istr2 = std::to_string(ilayer - 51) + " in z+";
    }
    histograms.h_clusternum_perlayer[ilayer] = ibook.book1D("totclusternum_layer_"+istr1,"total number of layer clusters for layer "+istr2,nintTotNClsperlay,minTotNClsperlay,maxTotNClsperlay);
    histograms.h_energyclustered_perlayer[ilayer] = ibook.book1D("energyclustered_perlayer"+istr1,"percent of total energy clustered by layer clusters over caloparticles energy for layer "+istr2,nintEneClperlay,minEneClperlay,maxEneClperlay);
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
    for (unsigned ilayer = 0; ilayer < 2*layers; ++ilayer) {
      auto istr1 = std::to_string(*it);
      auto istr2 = std::to_string(ilayer);
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
      histograms.h_cellsnum_perthickperlayer[istr] = ibook.book1D("cellsnum_perthick_perlayer_"+istr,"total number of cells for layer "+ istr3+" for thickness "+istr1,nintTotNcellsperthickperlayer,minTotNcellsperthickperlayer,maxTotNcellsperthickperlayer);
      //---
      histograms.h_distancetoseedcell_perthickperlayer[istr] = ibook.book1D("distancetoseedcell_perthickperlayer_"+istr,"distance of cluster cells to seed cell for layer "+ istr3+" for thickness "+istr1,nintDisToSeedperthickperlayer,minDisToSeedperthickperlayer,maxDisToSeedperthickperlayer);
      //---
      histograms.h_distancetoseedcell_perthickperlayer_eneweighted[istr] = ibook.book1D("distancetoseedcell_perthickperlayer_eneweighted_"+istr,"energy weighted distance of cluster cells to seed cell for layer "+ istr3+" for thickness "+istr1,nintDisToSeedperthickperlayerenewei,minDisToSeedperthickperlayerenewei,maxDisToSeedperthickperlayerenewei);
      //---
      histograms.h_distancetomaxcell_perthickperlayer[istr] = ibook.book1D("distancetomaxcell_perthickperlayer_"+istr,"distance of cluster cells to max cell for layer "+ istr3+" for thickness "+istr1,nintDisToMaxperthickperlayer,minDisToMaxperthickperlayer,maxDisToMaxperthickperlayer);
      //---
      histograms.h_distancetomaxcell_perthickperlayer_eneweighted[istr] = ibook.book1D("distancetomaxcell_perthickperlayer_eneweighted_"+istr,"energy weighted distance of cluster cells to max cell for layer "+ istr3+" for thickness "+istr1,nintDisToMaxperthickperlayerenewei,minDisToMaxperthickperlayerenewei,maxDisToMaxperthickperlayerenewei);
      //---
      histograms.h_distancebetseedandmaxcell_perthickperlayer[istr] = ibook.book1D("distancebetseedandmaxcell_perthickperlayer_"+istr,"distance of seed cell to max cell for layer "+ istr3+" for thickness "+istr1,nintDisSeedToMaxperthickperlayer,minDisSeedToMaxperthickperlayer,maxDisSeedToMaxperthickperlayer);
      //---
      histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer[istr] = ibook.book2D("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_"+istr,"distance of seed cell to max cell vs cluster energy for layer "+ istr3+" for thickness "+istr1,nintDisSeedToMaxperthickperlayer,minDisSeedToMaxperthickperlayer,maxDisSeedToMaxperthickperlayer,nintClEneperthickperlayer,minClEneperthickperlayer,maxClEneperthickperlayer);
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

void HGVHistoProducerAlgo::fill_generic_cluster_histos(const Histograms& histograms,
						       int count,
						       const reco::CaloClusterCollection &clusters, 
						       // const Density &densities,
						       std::vector<CaloParticle> const & cP,
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
    // std::cout << " seedid info " <<  seedid.rawId() << " " << seedid.det() << " " << HGCalDetId(seedid) << std::endl;
    const double seedx = recHitTools_->getPosition(seedid).x();
    const double seedy = recHitTools_->getPosition(seedid).y();
    DetId maxid = findmaxhit( clusters[layerclusterIndex] );
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
      // std::cout << " layerid " << layerid << " layer " << lay << std::endl;
      if (rh_detid.det() == DetId::Forward || rh_detid.det() == DetId::HGCalEE || rh_detid.det() == DetId::HGCalHSi){
	thickness = recHitTools_->getSiThickness(rh_detid);
      } else if (rh_detid.det() == DetId::HGCalHSc){
	thickness = -1;
      } else {
      	std::cout << "These are HGCal layer clusters, you shouldn't be here !!! " << layerid  << std::endl;
      	continue;
      }

      // thickness = (rh_detid.det() == DetId::Forward || rh_detid.det() == DetId::HGCalEE || rh_detid.det() == DetId::HGCalHSi) ? recHitTools_->getSiThickness(rh_detid) : -1;
      //Count here only once the layer cluster and save the combination thick_layerid
      std::string curistr = std::to_string( (int) thickness ) + "_" + std::to_string(layerid);
      if (cluslay){ tnlcpl[layerid]++; istr = curistr; cluslay = false; }

      if ( (thickness == 120.) && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhits120p++;}
      if ( (thickness == 120.) && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhits120m++;}
								     	  
      if ( (thickness == 200.) && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhits200p++;}
      if ( (thickness == 200.) && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhits200m++;}
								     	  
      if ( (thickness == 300.) && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhits300p++;}
      if ( (thickness == 300.) && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhits300m++;}
								     	  
      if ( (thickness == -1)   && (recHitTools_->zside(rh_detid) > 0. ) ) {nthhitsscintp++;}
      if ( (thickness == -1)   && (recHitTools_->zside(rh_detid) < 0. ) ) {nthhitsscintm++;}

      std::map<DetId,const HGCRecHit *>::const_iterator itcheck= hitMap_->find(rh_detid);
      if (itcheck == hitMap_->end()) {
	std::cout << " You shouldn't be here - Unable to find a hit " << rh_detid.rawId() << " " << rh_detid.det() << " " << HGCalDetId(rh_detid) << std::endl;
	continue;
      }

      const HGCRecHit *hit = itcheck->second;
      
      //Here for the per cell plots
      //----
      const double hit_x = recHitTools_->getPosition(rh_detid).x();
      const double hit_y = recHitTools_->getPosition(rh_detid).y();
      // std::cout << " hit_x " << hit_x << " hit_y " << hit_y << std::endl;
      double distancetoseed = distance(seedx, seedy, hit_x, hit_y);
      double distancetomax = distance(maxx, maxy, hit_x, hit_y);
      // std::cout << " DISTANCETOSEED " << distancetoseed << " distancetomax "<< distancetomax << " hit->energy() " << hit->energy() 
      // 		<< " seed times ene  " << distancetoseed*hit->energy() << " max times ene "<< distancetomax * hit->energy() << std::endl;
      // std::cout << curistr << std::endl;
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
      // std::map< DetId, float >::const_iterator dit = densities.find( rh_detid );
      // if ( dit == densities.end() ){
      // 	std::cout << " You shouldn't be here - Unable to find a density " << rh_detid.rawId() << " " << rh_detid.det() << " " << HGCalDetId(rh_detid) << std::endl;
      // 	continue;
      // }

      // // std::cout << " Detid " << dit->first << " Density " << dit->second << std::endl;
      // std::cout << " Density " << dit->second << std::endl;

      

      
    } // end of loop through hits and fractions

    //Check for simultaneously having hits of different kind. Checking at least two combinations is sufficient.
    if ( (nthhits120p != 0 && nthhits200p != 0  ) || (nthhits120p != 0 && nthhits300p != 0  ) || (nthhits120p != 0 && nthhitsscintp != 0  ) || 
	 (nthhits200p != 0 && nthhits300p != 0  ) || (nthhits200p != 0 && nthhitsscintp != 0  ) || (nthhits300p != 0 && nthhitsscintp != 0  ) ){
      // std::cout << "This cluster has hits of different kind: nthhits120 " << nthhits120 << " nthhits200 " 
      // 		<< nthhits200 << " nthhits300 " << nthhits300 << " nthhitsscint " << nthhitsscint <<std::endl;
      tnlcpthplus["mixed"]++;
    } else if ( (nthhits120p != 0 ||  nthhits200p != 0 || nthhits300p != 0 || nthhitsscintp != 0) )  {
      //This is a cluster with hits of one kind
      tnlcpthplus[std::to_string((int) thickness)]++;
    }
    if ( (nthhits120m != 0 && nthhits200m != 0  ) || (nthhits120m != 0 && nthhits300m != 0  ) || (nthhits120m != 0 && nthhitsscintm != 0  ) || 
	 (nthhits200m != 0 && nthhits300m != 0  ) || (nthhits200m != 0 && nthhitsscintm != 0  ) || (nthhits300m != 0 && nthhitsscintm != 0  ) ){
      // std::cout << "This cluster has hits of different kind: nthhits120 " << nthhits120 << " nthhits200 " 
      // 		<< nthhits200 << " nthhits300 " << nthhits300 << " nthhitsscint " << nthhitsscint <<std::endl;
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
    istr = std::to_string(thicknesses[ std::distance(bigamoth.begin(), bgth) ]) + "_" + std::to_string(layerid);

    // std::cout << istr << std::endl;

    //Here for the per cluster plots that need the thickness_layer info
    if (histograms.h_cellsnum_perthickperlayer.count(istr)){ histograms.h_cellsnum_perthickperlayer.at(istr).fill( hits_and_fractions.size() ); }
    
    //Now, with the distance between seed and max cell. 
    double distancebetseedandmax = distance(seedx, seedy, maxx, maxy);
    //The thickness_layer combination in this case will use the thickness of the seed as a convention. 
    std::string seedstr = std::to_string( (int) recHitTools_->getSiThickness(seedid) )+ "_" + std::to_string(layerid);
    std::cout << distancebetseedandmax << " cluster energy " << clusters[layerclusterIndex].energy() << " " << seedstr << std::endl;
    if (histograms.h_distancebetseedandmaxcell_perthickperlayer.count(seedstr)){ 
      histograms.h_distancebetseedandmaxcell_perthickperlayer.at(seedstr).fill( distancebetseedandmax ); 
    }
    if (histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer.count(seedstr)){ 
      histograms.h_distancebetseedandmaxcellvsclusterenergy_perthickperlayer.at(seedstr).fill( distancebetseedandmax , clusters[layerclusterIndex].energy() ); 
    }

    //Energy clustered per layer
    tecpl[layerid] = tecpl[layerid] + clusters[layerclusterIndex].energy();
    // std::cout << layerid << " " << clusters[layerclusterIndex].energy() << std::endl;
    ldbar[layerid] = ldbar[layerid] + clusters[layerclusterIndex].energy() * cummatbudg[(double) lay];

  }//end of loop through clusters of the event

  //For the mixed hits we want to know the percent
  // std::cout << std::accumulate(tnlcpl.begin(), tnlcpl.end(), 0) << ;

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
      // std::cout << "Total energy clustered for layer " << ilayer << ": " <<tecpl[ilayer] << std::endl;
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
      // std::cout << "Total energy clustered for layer " << ilayer << ": " <<tecpl[ilayer] << std::endl;
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

  std::cout << "Total energy clustered +z " << sumeneallcluspl << " and calo particles energy +z " << caloparteneplus << std::endl;
  std::cout << "Total energy clustered -z " << sumeneallclusmi << " and calo particles energy -z " << caloparteneminus << std::endl;

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


// void HGVHistoProducerAlgo::setRecHitTools(const hgcal::RecHitTools * recHitTools ) {
void HGVHistoProducerAlgo::setRecHitTools(std::shared_ptr<hgcal::RecHitTools> recHitTools) {
  recHitTools_ = recHitTools;
}

void HGVHistoProducerAlgo::fillHitMap(const HGCRecHitCollection & rechitsEE,
				      const HGCRecHitCollection & rechitsFH,
				      const HGCRecHitCollection & rechitsBH) {
  hitMap_->clear();
  for (const auto& hit : rechitsEE) {
    hitMap_->emplace_hint(hitMap_->end(), hit.detid(), &hit);
  }
 
  for (const auto& hit : rechitsFH) {
    hitMap_->emplace_hint(hitMap_->end(), hit.detid(), &hit);
  }
 
  for (const auto& hit : rechitsBH) {
    hitMap_->emplace_hint(hitMap_->end(), hit.detid(), &hit);
  }
 
}

DetId HGVHistoProducerAlgo::findmaxhit(const reco::CaloCluster & cluster) const {
  
  DetId themaxid;
  const std::vector<std::pair<DetId, float> > hits_and_fractions = cluster.hitsAndFractions();

  double maxene = 0.;
  for (std::vector<std::pair<DetId, float>>::const_iterator it_haf = hits_and_fractions.begin(); it_haf != hits_and_fractions.end(); ++it_haf) {
    DetId rh_detid = it_haf->first;

    std::map<DetId,const HGCRecHit *>::const_iterator itcheck= hitMap_->find(rh_detid);
    const HGCRecHit *hit = itcheck->second;

    if ( maxene < hit->energy() ){
      maxene = hit->energy();
      themaxid = rh_detid;
    }

  }

  return themaxid;
}


double HGVHistoProducerAlgo::getEta(double eta) const {
  if (useFabsEta) return fabs(eta);
  else return eta;
}

