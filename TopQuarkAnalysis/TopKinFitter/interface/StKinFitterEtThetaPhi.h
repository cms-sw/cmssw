#ifndef StKinFitterEtThetaPhi_h
#define StKinFitterEtThetaPhi_h

// includes for kinematic fit
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleESpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMCPInvSpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEScaledMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintM.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEtEtaPhi.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEtThetaPhi.hh"
//#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintMGaus.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintEp.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TKinFitter.hh"

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

// Root stuff
#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"


class StKinFitterEtThetaPhi {

  public:

    StKinFitterEtThetaPhi();
    StKinFitterEtThetaPhi(int,double,double,vector<int>);
    ~StKinFitterEtThetaPhi();

    StEvtSolution addKinFitInfo(StEvtSolution * asol);

  private:

    void setupFitter();

  private:

    TKinFitter * theFitter;

    TFitParticleEMomDev      	   * fitBottom;
    TFitParticleEMomDev      	   * fitLight;
    TFitParticleEtThetaPhi      * fitLepl;
    TFitParticleEtThetaPhi      * fitLepn;

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
