#ifndef TopKinFitter_StKinFitterEtEtaPhi_h
#define TopKinFitter_StKinFitterEtEtaPhi_h

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
