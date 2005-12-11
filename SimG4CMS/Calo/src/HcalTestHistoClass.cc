///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoClass.cc
// Description: Histogram handling class in HcalTestAnalysis (HcalTest)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalTestHistoClass.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"

#include <iostream>
#include <cmath>

void HcalTestHistoClass::setCounters(){
  nLayers   = 0;
  nHits     = 0;
  nQIE      = 0;
  nTowerQIE = 0;
  nGroupQIE = 0;
}

void HcalTestHistoClass::fillLayers(double* edepl, double edepHO,
				    double edepHBHE, double* muxy) {

  nLayers = 0;
  layers.resize(nLayersMAX);
  for (int i = 0; i < 20; i++) {
    double ed  = 0.001*edepl[i];
    if (verbosity > 2)
      std::cout << "HcalTestHistoClass:: fillLayer: nLayers, ed " << i 
		<< " "  << ed  << std::endl;
    if (nLayers < nLayersMAX) {
      layers[i].e = ed;
      layers[i].muDist = muxy[i];
      nLayers++;
    }
  }

  eHO   = edepHO;
  eHBHE = edepHBHE;
}

void HcalTestHistoClass::fillHits(std::vector<CaloHit> hitcache) {

  int nHit = hitcache.size();
  int hit  = 0;
  int i;
  std::vector<CaloHit>::iterator itr;
  std::vector<CaloHit*> lhits(nHit);
  for (i = 0, itr = hitcache.begin(); itr != hitcache.end(); i++, itr++) {
    uint32_t unitID=itr->id();
    int   subdet, zside, group, ieta, iphi, lay;
    HcalTestNumberingScheme::unpackHcalIndex(unitID,subdet,zside,group,
					     ieta,iphi,lay);
    subdet = itr->det();
    lay    = itr->layer();
    group  = (subdet&15)<<20;
    group += ((lay-1)&31)<<15;
    group += (zside&1)<<14;
    group += (ieta&127)<<7;
    group += (iphi&127);
    itr->setId(group);
    lhits[i] = &hitcache[i];
    if (verbosity > 2) {
      std::cout << "HcalTestHistoClass::fillHits:Original " << i << " " 
		<< hitcache[i] << std::endl;
      std::cout << "HcalTestHistoClass::fillHits:Copied   " << i << " " 
		<< *lhits[i]    << std::endl;
    }
  }
  sort(lhits.begin(),lhits.end(),CaloHitIdMore());
  std::vector<CaloHit*>::iterator k1, k2;
  if (verbosity > 2) {
    for (i = 0, k1 = lhits.begin(); k1 != lhits.end(); i++, k1++)
      std::cout << "HcalTestHistoClass::fillHits:Sorted " << i << " " 
		<< **k1 << std::endl;
  }
  hits.resize(lhits.size());
  for (i = 0, k1 = lhits.begin(); k1 != lhits.end(); i++, k1++) {
    double       ehit  = (**k1).e();
    double       jitter= (**k1).t();
    unsigned int unitID= (**k1).id();
    int          jump  = 0;
    if (verbosity > 2) 
      std::cout << "HcalTestHistoClass::fillHits:Start " << i << " U/T/E 0x"
		<< std::hex << unitID << std::dec << " "  << jitter << " " 
		<< ehit;
    for (k2 = k1+1; k2 != lhits.end() && (jitter-(**k2).t())<1. &&
	   (jitter-(**k2).t())>-1. && unitID==(**k2).id(); k2++) {
      ehit += (**k2).e();
      if (verbosity > 2) std::cout << " + " << (**k2).e();
      jump++;
    }
    if (verbosity > 2) 
      std::cout << " = " << ehit << " in " << jump << std::endl;

    float eta   = itr->eta();
    float phi   = itr->phi();
    float t     = itr->t();
    int lay    = (unitID>>15)&31 + 1;

    hits[nHits].layer = lay;
    hits[nHits].id = unitID;
    hits[nHits].eta = eta;
    hits[nHits].phi = phi;
    hits[nHits].e = ehit;
    hits[nHits].t = t;
    hits[nHits].jitter = jitter;
    nHits++;
    hit++;

    int subdet = (unitID>>20)&15;
    int zside  = (unitID>>14)&1;
    int ieta   = (unitID>>7)&127;
    int iphi   = (unitID)&127;
    if (verbosity > 1) 
      std::cout << "HcalTestHistoClass::fillHits:Hit " << hit << " " << i 
		<< " ID 0x" << std::hex << unitID << std::dec << " " << subdet 
		<< " " << lay << " " << zside << " " << ieta << " " << iphi 
		<< " Time " << jitter << " E " << ehit << std::endl;

    i  += jump;
    k1 += jump;
  }

  if (verbosity > 1) 
    std::cout << "HcalTestHistoClass::fillHits called with " << nHit 
	      << " hits" << " and writes out " << nHits << '(' << hit 
	      << ") hits" << std::endl;

}

void HcalTestHistoClass::fillQie (int id, double esimtot, double eqietot, 
				  int nGroup, std::vector<double> longs,
				  std::vector<double> longq, int nTower,
				  std::vector<double> latphi, 
				  std::vector<double> latfs, 
				  std::vector<double> latfq) {

  nGroupQIE = 0;
  nTowerQIE = 0;
  
  if (id>=0 && id<4) {
    unsigned int qiesiz = (unsigned int)(id+1);
    if (qie.size()<qiesiz) {
      qie.resize(qiesiz);     
    }

    qie[id].sim=esimtot;
    qie[id].qie=eqietot;
    qie[id].id = id;
    nQIE++;
    
    if (verbosity > 2) 
      std::cout << "HcalTestHistoClass::fillQie: id, esimtot, eqietot = "
		<< id << " " << esimtot << " " << eqietot  << std::endl;
    
    
    for (int i=0; i<nGroup; i++) {
      if (verbosity > 2) 
	std::cout << "HcalTestHistoClass::fillQie: id, nGroupQIE, longs, "
		  << "longq = " << id << " " << nGroupQIE << " " << longs[i]
		  << " " << longq[i] << std::endl;
      qie[id].lngs.push_back(longs[i]);
      qie[id].lngq.push_back(longq[i]);
      nGroupQIE++;
    }

    for (int i=0; i<nTower; i++) {
      int tow = (int)latphi[i];
      if (verbosity > 2) 
	std::cout << "HcalTestHistoClass::fillQie: id, nTowerQIE, tower, "
		  << "latfs, latfq = " << id << " " << nTowerQIE << " " 
		  << tow << " " << latfs[i] << " " << latfq[i] << std::endl;
      qie[id].lats.push_back(latfs[i]);
      qie[id].latq.push_back(latfq[i]);
      qie[id].tow.push_back(tow);
      nTowerQIE++;
    }
  }
  if (verbosity > 1) 
    std::cout << "HcalTestHistoClass::fillQie: Called with ID " << id
	      << " nQIE " << nQIE << " nGroup " << nGroupQIE << " nTower "
	      << nTowerQIE << std::endl;
}

