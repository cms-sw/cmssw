///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoClass.h
// Histogram handling class for analysis in HCALTest
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalTestHistoClass_H
#define HcalTestHistoClass_H

#include "SimG4CMS/Calo/interface/CaloHit.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <boost/cstdint.hpp>
#include <string>
#include <vector>

class HcalTestHistoClass : public TObject {

public: 

  HcalTestHistoClass(int, std::string);
  virtual ~HcalTestHistoClass();

  void setCounters();
  void fillLayers (double el[], double ho, double hbhe, double muxy[]);
  void fillHits   (std::vector<CaloHit>);
  void fillQie    (int id, double esimtot, double eqietot, int nGroup,
		   std::vector<double> longs,  std::vector<double> longq,
		   int nTower, std::vector<double> latphi, 
		   std::vector<double> latfs, std::vector<double> latfq);
  void fillTree   ();

private:

  TTree *tree;
  TFile *froot;

  int     verbosity;
  const static Int_t nLayersMAX = 20;
  Int_t   nLayers;
  Float_t eLayer[nLayersMAX], muDist[nLayersMAX], eHO, eHBHE;
 
  const static Int_t nHitsMAX = 15000;
  Int_t   nHits, layerHits[nHitsMAX], idHits[nHitsMAX];
  Float_t etaHits[nHitsMAX], phiHits[nHitsMAX], eHits[nHitsMAX];
  Float_t tHits[nHitsMAX], jitterHits[nHitsMAX];
 
  const static Int_t nQIEMAX = 4, nTowerQIEMAX = 100, nGroupQIEMAX = 20;
  Int_t   nQIE, nTowerQIE, nGroupQIE;
  Float_t simQIE[nQIEMAX],  qieQIE[nQIEMAX];
  Float_t latsQIE[nQIEMAX][nTowerQIEMAX], latqQIE[nQIEMAX][nTowerQIEMAX];
  Float_t lngsQIE[nQIEMAX][nGroupQIEMAX], lngqQIE[nQIEMAX][nGroupQIEMAX];
  Int_t   idQIE[nQIEMAX], towQIE[nQIEMAX][nTowerQIEMAX];

};

#endif
