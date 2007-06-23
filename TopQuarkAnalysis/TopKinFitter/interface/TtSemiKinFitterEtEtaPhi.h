#ifndef TopKinFitter_TtSemiKinFitterEtEtaPhi_h
#define TopKinFitter_TtSemiKinFitterEtEtaPhi_h

// includes for kinematic fit
#include "PhysicsTools/KinFitter/interface/TFitParticleESpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCPInvSpher.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEScaledMomDev.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
//#include "PhysicsTools/KinFitter/interface/TFitConstraintMGaus.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"

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
