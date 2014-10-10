#ifndef BINS_H
#define BINS_H

#ifndef M_PI 
#define M_PI 3.141592653589793
#endif


Float_t *getBins(Int_t nBins, Float_t fMin, Float_t fMax){
  Float_t *afBins = new Float_t[nBins + 1];

  for(int i = 0; i<=nBins; i++){
    afBins[i] = fMin + (fMax - fMin)*i/nBins;
  }

  return afBins;
}


//_______________________________________________
Int_t         nDcaBins  = 100;
Float_t       fDcaMax   = .1; 
Float_t       fDcaMin   = -.1; 
Float_t      *afDcaBins = 0;

Float_t       fEtaMax =  2.5;          
Float_t       fEtaMin = -2.5;          
Int_t         nEtaBins = 50;
Float_t      *afEtaBins = 0;

Float_t       fYMax =  2.5;          
Float_t       fYMin = -2.5;          
Int_t         nYBins = 50;
Float_t      *afYBins = 0;


Int_t         nMassBins  = 100;
Float_t       fMassMax   = 3.5; 
Float_t       fMassMin   = 2.5; 
Float_t      *afMassBins = 0;

Float_t       fPhiMax =  M_PI;       
Float_t       fPhiMin = -M_PI;       
Int_t         nPhiBins = 15;
Float_t      *afPhiBins = 0;

Float_t       fThetaMax =  M_PI;       
Float_t       fThetaMin = -M_PI;       
Int_t         nThetaBins = 15;
Float_t      *afThetaBins = 0;

Float_t       fPtMax  = 20.;         
Float_t       fPtMin  =  0.;         
Int_t         nPtBins  = 40;
Float_t      *afPtBins = 0;

Float_t       fNormMin = -1.;
Float_t       fNormMax =  1.;
Int_t         nNormBins  = 200;
Float_t      *afNormBins = 0;

Float_t       fIntMin = 0.5;
Float_t       fIntMax = 1.1;
Int_t         nIntBins = 100;
Float_t      *afIntBins = 0;

Int_t nDiPtResBins = 25;
Float_t afDiPtResBins[26] = {20.,24.,28.,
			     32.,36.,
			     40.,44.,48.,
			     52.,56.,
			     60.,64.,68.,
			     72.,76.,
			     80.,84.,88.,
			     92.,96.,
			     100.,110.,120.,130.,140.,150.};

Int_t nPtResBins = 38;
Float_t afPtResBins[39] = {2.,7.,
			   10.,15.,
			   22.,26.,
			   30.,34.,38.,
			   42.,46.,
			   50.,52.,54.,56.,58.,
			   60.,62.,64.,66.,68.,
			   70.,72.,74.,76.,78.,
			   80.,82.,84.,86.,88.,
			   90.,95.,
			   100., 110., 120., 130., 140., 150.};

Int_t nTypeBins     = 7;
Float_t fTypeMax    = 6.5;
Float_t *afTypeBins = 0;

// must call this before using bins!
void initBins()
{
  afDcaBins  = getBins(nDcaBins, fDcaMin, fDcaMax);
  afEtaBins  = getBins(nEtaBins, fEtaMin, fEtaMax);
  afIntBins  = getBins(nIntBins, fIntMin, fIntMax);
  afMassBins = getBins(nMassBins, fMassMin, fMassMax);
  afNormBins = getBins(nNormBins, fNormMin, fNormMax);
  afPhiBins  = getBins(nPhiBins, fPhiMin, fPhiMax);
  afPtBins   = getBins(nPtBins, fPtMin, fPtMax);
  afTypeBins = getBins(nTypeBins, -0.5, fTypeMax);
  afThetaBins  = getBins(nThetaBins, fThetaMin, fThetaMax);
  afYBins  = getBins(nYBins, fYMin, fYMax);
}

#endif
