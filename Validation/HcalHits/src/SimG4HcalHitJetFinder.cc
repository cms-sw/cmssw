///////////////////////////////////////////////////////////////////////////////
// File: SimG4HcalHitJetFinder.cc
// Description: Jet finder class for SimG4HcalValidation
///////////////////////////////////////////////////////////////////////////////
#include "Validation/HcalHits/interface/SimG4HcalHitJetFinder.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <iostream>
#include <cmath>

SimG4HcalHitJetFinder::SimG4HcalHitJetFinder(int iv, double cone):
  verbosity(iv), jetcone(cone) {}

SimG4HcalHitJetFinder::~SimG4HcalHitJetFinder() {
  if (verbosity > 0)
    std::cout << "SimG4HcalHitJetFinder:: Deleting" << std::endl;
}

void SimG4HcalHitJetFinder::setCone(double cone) { 
  jetcone = cone;
}   

void SimG4HcalHitJetFinder::setInput(std::vector<CaloHit>* hhit) { 
  input   = * hhit;
}

std::vector<SimG4HcalHitCluster>* SimG4HcalHitJetFinder::getClusters(bool hcal_only){
  clusvector.erase(clusvector.begin(),clusvector.end()); 
  if (input.size() == 0) { 
    return &clusvector;
  }

  std::vector<CaloHit>::iterator itr;
  if (verbosity > 1) {
    for (itr = input.begin(); itr != input.end(); itr++) {
      std::cout << " getClusters_1 - input :  e " << itr->e() << "  eta " 
		<< itr->eta() << "  phi " << itr->phi() << "  subdet " 
		<< itr->det() << std::endl;
    }
  }

  sort(input.begin(),input.end()); // sort input in descending order
 
  if (verbosity > 1) {
    for (itr = input.begin(); itr != input.end(); itr++) {
      std::cout << " getClusters_2 - input :  e " << itr->e() << "  eta " 
		<< itr->eta() << "  phi " << itr->phi() << "  subdet " 
		<< itr->det() << std::endl;
    }
  }

  std::vector<SimG4HcalHitCluster> temp;   // dummy container for clusters

  //  first input hit -> first cluster

  CaloHit             hit;
  SimG4HcalHitCluster cluster;

  std::vector<CaloHit>::iterator itr_hits;

  int j, first_seed = 0;
  for (j=0, itr_hits = input.begin(); itr_hits != input.end();
       j++, itr_hits++) {
    int h_type = itr_hits->det(); //if desired HCAL hits (only) clusterfinding
    if (((h_type == static_cast<int>(HcalBarrel) ||
	  h_type == static_cast<int>(HcalEndcap) ||
	  h_type == static_cast<int>(HcalForward)) && hcal_only) || 
	(!hcal_only)) {
      cluster += input[j];
      if (verbosity > 2) 
	std::cout << " First seed hit .................." << std::endl
		  << (*itr_hits) << std::endl;
      first_seed = j;
      break;
    }
  }
  
  temp.push_back(cluster);
  
  std::vector<SimG4HcalHitCluster>::iterator itr_clus;

  for (j=0, itr_hits = input.begin(); itr_hits != input.end();
       j++, itr_hits++) {
    int h_type = itr_hits->det(); //if desired HCAL hits (only) clusterfinding
    if ((((h_type == static_cast<int>(HcalBarrel) ||
	   h_type == static_cast<int>(HcalEndcap) ||
	   h_type == static_cast<int>(HcalForward)) && hcal_only) || 
	 (!hcal_only)) && (j != first_seed)) {
      if (verbosity > 2) 
	std::cout << " ........... Consider hit .................." 
		  << std::endl << (*itr_hits) << std::endl;
      
      int incl = 0; // if the hit is included in one of clusters
      
      int iclus;  
      for (itr_clus = temp.begin(), iclus = 0; itr_clus != temp.end();
	   itr_clus++, iclus++) { 
	
	if (verbosity > 2)
	  std::cout << "=======> Cluster " << iclus << std::endl
		    << (*itr_clus) << std::endl;
	
	double d = rDist(&(*itr_clus), &(*itr_hits));
	if (d < jetcone) {
	  if (verbosity > 2) std::cout << " -> associated ... " << std::endl;
	  temp[iclus] += *itr_hits;
	  incl = 1;
	  break;  
	}
      }
      
      // to here jumps "break"
      if (incl == 0) {
	SimG4HcalHitCluster cl;
	cl += *itr_hits;
	temp.push_back(cl);
	if (verbosity > 2)
	  std::cout << std::endl << " ************ NEW CLUSTER !" << std::endl
		    << cl << std::endl;
      }
    }
  }

  clusvector = temp;
  return &clusvector;
}

double SimG4HcalHitJetFinder::rDist(const SimG4HcalHitCluster* cluster, 
				    const CaloHit* hit) const {

  double etac = cluster->eta();
  double phic = cluster->phi();
  
  double etah = hit->eta();
  double phih = hit->phi();

  return rDist(etac, phic, etah, phih);
}


double SimG4HcalHitJetFinder::rDist(const double etac,const double phic,
				    const double etah,const double phih) const{

  double delta_eta = etac - etah;
  double delta_phi = phic - phih;

  if (phic < phih)      delta_phi = phih - phic;
  if (delta_phi > M_PI) delta_phi = 2*M_PI - delta_phi;

  if (verbosity > 1)
    std::cout << std::endl
	      << " Clus. eta, phi = " << etac << " " << phic << std::endl 
	      << " hit   eta, phi = " << etah << " " << phih << " rDist = "
	      << sqrt(delta_eta * delta_eta + delta_phi * delta_phi)
	      << std::endl;

  return sqrt(delta_eta * delta_eta + delta_phi * delta_phi);
}

