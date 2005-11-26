///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoClass.cc
// Description: Histogram handling class in HcalTestAnalysis (HCALTest)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalTestHistoClass.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <cmath>

HcalTestHistoClass::HcalTestHistoClass(int iv, std::string file) : 
  verbosity(iv) {

  const char * cfile = file.c_str();
  froot = new TFile(cfile, "RECREATE");
  if (froot == 0) {
    if (verbosity > 0)
      std::cout << "HcalTestHistoClass:: opening file " << cfile << " failed"
		<< std::endl;
    throw cms::Exception("Unknown", "HcalTestHistoClass")
      << "Fails in opening output file " << cfile << "\n";
  } else {
    if (verbosity > 0)
      std::cout << "HcalTestHistoClass:: opens file " << cfile << std::endl;
  }
  froot->SetCompressionLevel(2);
  tree  = new TTree("HcalTestAnalysis", "Root Tree for HcalTestAnalysis");
  if (verbosity > 0)
    std::cout << std::endl << "===>>>  Start booking user Root tree" 
	      << std::endl;

  // Layers and Depthes =======================================================
   
  tree->Branch("nLayers",     &nLayers,   "nLayers/I");
  tree->Branch("eLayer",      &eLayer,    "eLayer[nLayers]/F");
  tree->Branch("muDist",      &muDist,    "muDist[nLayers]/F");
  tree->Branch("eHO",         &eHO,       "eHO/F");
  tree->Branch("eHBHE",       &eHBHE,     "eHBHE/F");
   
  // All Hits properties grid
  tree->Branch("nHits",      &nHits,      "nHits/I");
  tree->Branch("layerHits",  &layerHits,  "layerHits[nHits]/I");
  tree->Branch("etaHits",    &etaHits,    "etaHits[nHits]/F");
  tree->Branch("phiHits",    &phiHits,    "phiHits[nHits]/F");
  tree->Branch("eHits",      &eHits,      "eHits[nHits]/F");
  tree->Branch("tHits",      &tHits,      "tHits[nHits]/F");
  tree->Branch("idHits",     &idHits,     "idHits[nHits]/I");
  tree->Branch("jitterHits", &jitterHits, "jitterHits[nHits]/F");
 
  // QIE properties ===========================================================
  tree->Branch("nQIE",         &nQIE,        "nQIE/I");
  tree->Branch("simQIE ",      &simQIE,      "simQIE[nQIE]/F");
  tree->Branch("qieQIE",       &qieQIE,      "qieQIE[nQIE]/F");
  tree->Branch("idQIE",        &idQIE,       "idQIE[nQIE]/I");
  tree->Branch("nTowerQIE",    &nTowerQIE,   "nTowerQIE/I");
  tree->Branch("latsQIE",      &latsQIE,     "latsQIE[nQIE][nTowerQIE]/F");
  tree->Branch("latqQIE",      &latqQIE,     "latqQIE[nQIE][nTowerQIE]/F");
  tree->Branch("towQIE",       &towQIE,      "towQIE[nQIE][nTowerQIE]/F");
  tree->Branch("nGroupQIE",    &nGroupQIE,   "nGroupQIE/I");
  tree->Branch("lngsQIE",      &lngsQIE,     "lngsQIE[nQIE][nGroupQIE]/F");
  tree->Branch("lngqQIE",      &lngqQIE,     "lngqQIE[nQIE][nGroupQIE]/F");

  if (verbosity > 0) {
    std::cout << std::endl << "===>>> Done booking user histograms & Ntuples " 
	      << std::endl;
  }
}

HcalTestHistoClass::~HcalTestHistoClass() {

  if (verbosity > 0) 
    std::cout << "========================================================" 
	      << std::endl
	      << "=== HcalTestHistoClass: Start writing user histograms ===" 
	      << std::endl;

  // Save the trees
  tree->Print();
  if (verbosity > 1) 
    std::cout << "HcalTestHistoClass: Pointer to tree " << tree 
	      << " and that to root file " << froot << std::endl;
  //  froot->Write("", TObject::kOverwrite);
  //  froot->Close();
  delete tree;
  delete froot;
  
  if (verbosity > 0) 
    std::cout << std::endl << "HcalTestHistoClass: End writing user histograms"
	      << std::endl;
}

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
  for (int i = 0; i < 20; i++) {
    double ed  = 0.001*edepl[i];
    if (verbosity > 2)
      std::cout << "HcalTestHistoClass:: fillLayer: nLayers, ed " << i 
		<< " "  << ed  << std::endl;
    if (nLayers < nLayersMAX) {
      eLayer[nLayers] = (Float_t)ed;
      muDist[nLayers] = (Float_t)muxy[i];
      nLayers++;
    }
  }

  eHO   = (Float_t)edepHO;
  eHBHE = (Float_t)edepHBHE;
}

void HcalTestHistoClass::fillHits(std::vector<CaloHit> hitcache) {

  int nHit = hitcache.size();
  int hit  = 0;
  int i;
  std::vector<CaloHit>::iterator itr;
  std::vector<CaloHit*> hits(nHit);
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
    hits[i] = &hitcache[i];
    if (verbosity > 2) {
      std::cout << "HcalTestHistoClass::fillHits:Original " << i << " " 
		<< hitcache[i] << std::endl;
      std::cout << "HcalTestHistoClass::fillHits:Copied   " << i << " " 
		<< *hits[i]    << std::endl;
    }
  }
  sort(hits.begin(),hits.end(),CaloHitIdMore());
  std::vector<CaloHit*>::iterator k1, k2;
  if (verbosity > 2) {
    for (i = 0, k1 = hits.begin(); k1 != hits.end(); i++, k1++)
      std::cout << "HcalTestHistoClass::fillHits:Sorted " << i << " " 
		<< **k1 << std::endl;
  }
  for (i = 0, k1 = hits.begin(); k1 != hits.end(); i++, k1++) {
    double       ehit  = (**k1).e();
    double       jitter= (**k1).t();
    unsigned int unitID= (**k1).id();
    int          jump  = 0;
    if (verbosity > 2) 
      std::cout << "HcalTestHistoClass::fillHits:Start " << i << " U/T/E 0x"
		<< std::hex << unitID << std::dec << " "  << jitter << " " 
		<< ehit;
    for (k2 = k1+1; k2 != hits.end() && (jitter-(**k2).t())<1. &&
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
    if (nHits < nHitsMAX) {
      layerHits[nHits]  = (Int_t)lay;
      idHits[nHits]     = (Int_t)unitID;
      etaHits[nHits]    = (Float_t)eta;
      phiHits[nHits]    = (Float_t)phi;
      eHits[nHits]      = (Float_t)ehit;
      tHits[nHits]      = (Float_t)t;
      jitterHits[nHits] = (Float_t)jitter;
      nHits++;
    }
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
    simQIE[id] = (Float_t)esimtot;
    qieQIE[id] = (Float_t)eqietot;
    idQIE[id]  = (Int_t)id;
    nQIE++;
 
    if (verbosity > 2) 
      std::cout << "HcalTestHistoClass::fillQie: id, esimtot, eqietot = "
		<< id << " " << esimtot << " " << eqietot  << std::endl;

    for (int i=0; i<nGroup; i++) {
      if (verbosity > 2) 
	std::cout << "HcalTestHistoClass::fillQie: id, nGroupQIE, longs, "
		  << "longq = " << id << " " << nGroupQIE << " " << longs[i]
		  << " " << longq[i] << std::endl;
      if (nGroupQIE < nGroupQIEMAX) {
	lngsQIE[id][nGroupQIE] = (Float_t)longs[i];
	lngqQIE[id][nGroupQIE] = (Float_t)longq[i];
	nGroupQIE++;
      }
    }

    for (int i=0; i<nTower; i++) {
      int tow = (int)latphi[i];
      if (verbosity > 2) 
	std::cout << "HcalTestHistoClass::fillQie: id, nTowerQIE, tower, "
		  << "latfs, latfq = " << id << " " << nTowerQIE << " " 
		  << tow << " " << latfs[i] << " " << latfq[i] << std::endl;
      if (nTowerQIE < nTowerQIEMAX) {
	latsQIE[id][nTowerQIE] = (Float_t)latfs[i];
	latqQIE[id][nTowerQIE] = (Float_t)latfq[i];
	towQIE[id][nTowerQIE]  = (Int_t)tow;
	nTowerQIE++;
      }
    }
  }
  if (verbosity > 1) 
    std::cout << "HcalTestHistoClass::fillQie: Called with ID " << id
	      << " nQIE " << nQIE << " nGroup " << nGroupQIE << " nTower "
	      << nTowerQIE << std::endl;
}

void HcalTestHistoClass::fillTree() {

  if (verbosity > 1) 
    std::cout << "HcalTestHistoClass::fillTree called with nLayers = " 
	      << nLayers << " nHits = " << nHits << " nQIE = " << nQIE
	      << " nTowerQIE = " << nTowerQIE << " nGroupQIE = " << nGroupQIE 
	      << " tree pointer = " << tree << std::endl;
  //  tree->Fill();
}
