#ifndef TauAnalysis_SVfitStandalone_LikelihoodFunctions_h
#define TauAnalysis_SVfitStandalone_LikelihoodFunctions_h

#include "TMatrixD.h"
#include "TH1.h"

/**
   \class   probMET LikelihoodFunctions.h "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"
   
   \brief   Likelihood for MET

   Likelihood for MET. Input parameters are:

    pMETX  : difference between reconstructed MET and fitted MET in x
    pMETY  : difference between reconstructed MET and fitted MET in y
    covDet : determinant of the covariance matrix of the measured MET
    covInv : 2-dim inverted covariance matrix for measured MET (to be 
             determined from MET significance algorithm)
    power  : additional power to enhance the nll term
*/
double probMET(double dMETX, double dMETY, double covDet, const TMatrixD& covInv, double power = 1., bool verbose = false);

/**
   \class   probTauToLepPhaseSpace LikelihoodFunctions.h "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"
   
   \brief   Matrix Element for leptonic tau decays. Input parameters are:

   \var decayAngle : decay angle in the restframe of the tau lepton decay
   \var visMass : measured visible mass
   \var nunuMass : fitted neutrino mass

   The parameter tauLeptonMass2 is the mass of the tau lepton squared as defined in svFitStandaloneAuxFunctions.h    

     NOTE: The formulas taken from the paper
             "Tau polarization and its correlations as a probe of new physics",
             B.K. Bullock, K. Hagiwara and A.D. Martin,
             Nucl. Phys. B395 (1993) 499.

*/
double probTauToLepMatrixElement(double decayAngle, double nunuMass, double visMass, double x, bool applySinTheta, bool verbose = false);

/**
   \class   probTauToHadPhaseSpace LikelihoodFunctions.h "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"
   
   \brief   Likelihood for a two body tau decay into two hadrons

   Likelihood for a two body tau decay into two hadrons. Input parameters is:

    decayAngle : decay angle in the restframe of the tau lepton decay
*/
double probTauToHadPhaseSpace(double decayAngle, double nunuMass, double visMass, double x, bool applySinTheta, bool verbose = false);

/**
   \class   probVisMass LikelihoodFunctions.h "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"
   
   \brief   Likelihood for producing system of given mass in hadronic tau decay

   Likelihood for hadronic tau decay to produce visible decay products of true mass (visMass)

    lutVisMass : histograms that parametrize the mass distribution of the visible decay products produced in hadronic tau decays on generator level
*/
double probVisMass(double visMass, const TH1* lutVisMass, bool verbose = false);

/**
   \class   probVisMassShift, probVisPtShift LikelihoodFunctions.h "TauAnalysis/SVfitStandalone/interface/LikelihoodFunctions.h"
   
   \brief   Resolution on Pt and mass of hadronic taus

   Likelihood for a hadronic tau of true Pt and mass (visMass, visPt)
   to be reconstructed with Pt and mass of (visMass + deltaVisMass, recTauPtDivGenTauPt*visPt)

    lutVisMassRes, lutVisPtRes : histograms that parametrize the Pt and mass resolution for hadronic taus
*/
double probVisMassShift(double deltaVisMass, const TH1* lutVisMassRes, bool verbose = false);
double probVisPtShift(double recTauPtDivGenTauPt, const TH1* lutVisPtRes, bool verbose = false);

#endif
