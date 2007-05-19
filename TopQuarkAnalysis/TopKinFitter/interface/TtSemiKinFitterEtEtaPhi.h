#ifndef TtSemiKinFitterEtEtaPhi_h
#define TtSemiKinFitterEtEtaPhi_h

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

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

// Root stuff
#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"


class TtSemiKinFitterEtEtaPhi {

  public:

    TtSemiKinFitterEtEtaPhi();
    TtSemiKinFitterEtEtaPhi(int,double,double,vector<int>);
    ~TtSemiKinFitterEtEtaPhi();

    TtSemiEvtSolution addKinFitInfo(TtSemiEvtSolution * asol);

  private:

    void setupFitter();

  private:

    TKinFitter * theFitter;

    TFitParticleEtEtaPhi      * fitHadb;
    TFitParticleEtEtaPhi      * fitHadp;
    TFitParticleEtEtaPhi      * fitHadq;
    TFitParticleEtEtaPhi      * fitLepb;
    TFitParticleEtEtaPhi      * fitLepl;
    TFitParticleEtEtaPhi      * fitLepn;

    TFitConstraintM  * cons1;
    TFitConstraintM  * cons2;
    TFitConstraintM  * cons3;
    TFitConstraintM  * cons4;
    TFitConstraintM  * cons5;
    
    // parameters
    bool debugOn;
    bool doNeutrinoResol;
    int maxNrIter;
    double maxDeltaS;
    double maxF;  
    vector<int> constraints;
    vector<double> translateCovM(TMatrixD &);
};


#endif
