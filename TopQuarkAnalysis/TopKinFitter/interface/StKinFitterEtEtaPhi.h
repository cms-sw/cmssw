#ifndef StKinFitterEtEtaPhi_h
#define StKinFitterEtEtaPhi_h

// includes for kinematic fit
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleESpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMCPInvSpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEScaledMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintM.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEtEtaPhi.hh"
//#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintMGaus.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintEp.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TKinFitter.hh"

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

// Root stuff
#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"


class StKinFitterEtEtaPhi {

  public:

    StKinFitterEtEtaPhi();
    StKinFitterEtEtaPhi(int,double,double,vector<int>);
    ~StKinFitterEtEtaPhi();

    StEvtSolution addKinFitInfo(StEvtSolution * asol);

  private:

    void setupFitter();

  private:

    TKinFitter * theFitter;

    TFitParticleEMomDev      	   * fitBottom;
    TFitParticleEMomDev      	   * fitLight;
    TFitParticleEtEtaPhi      * fitLepl;
    TFitParticleEtEtaPhi      * fitLepn;

    TFitConstraintM  * cons1;
    TFitConstraintM  * cons2;
    TFitConstraintM  * cons3;
    
    // parameters
    bool debugOn;
    bool doNeutrinoResol;
    int maxNrIter;
    double maxDeltaS;
    double maxF;  
    vector<int> constraints;
};


#endif
