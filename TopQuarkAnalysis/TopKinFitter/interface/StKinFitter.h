//
// $Id: StKinFitter.h,v 1.1 2007/09/19 23:05:31 lowette Exp $
//

#ifndef TopKinFitter_StKinFitter_h
#define TopKinFitter_StKinFitter_h

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"

#include <vector>

class TKinFitter;
class TAbsFitParticle;
class TFitConstraintM;

class StKinFitter {

  public:

    enum Parametrization { EMom, EtEtaPhi, EtThetaPhi };

  public:

    StKinFitter();
    StKinFitter(int jetParam, int lepParam, int metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
    StKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
    ~StKinFitter();

    StEvtSolution addKinFitInfo(StEvtSolution * asol);

  private:

    void setupFitter();
    std::vector<float> translateCovM(TMatrixD &);

  private:

    // the kinematic fitter
    TKinFitter * theFitter_;
    // the particles that enter the kinematic fit
    TAbsFitParticle * fitBottom_;
    TAbsFitParticle * fitLight_;
    TAbsFitParticle * fitLepton_;
    TAbsFitParticle * fitNeutrino_;
    // the constraints on the fit
    TFitConstraintM  * cons1_;
    TFitConstraintM  * cons2_;
    TFitConstraintM  * cons3_;
    // other parameters
    Parametrization jetParam_, lepParam_, metParam_;
    bool doNeutrinoResol_;
    int maxNrIter_;
    double maxDeltaS_;
    double maxF_;
    std::vector<int> constraints_;
};


#endif
