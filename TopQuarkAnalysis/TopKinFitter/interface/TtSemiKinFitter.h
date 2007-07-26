//
// $Id: TtSemiKinFitter.h,v 1.1 2007/07/19 00:32:12 lowette Exp $
//

#ifndef TopKinFitter_TtSemiKinFitter_h
#define TopKinFitter_TtSemiKinFitter_h

#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"

#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"

#include <vector>


class TKinFitter;
class TAbsFitParticle;
class TFitConstraintM;


class TtSemiKinFitter {

  public:

    enum Parametrization { EMom, EtEtaPhi, EtThetaPhi };

  public:

    TtSemiKinFitter();
    TtSemiKinFitter(int jetParam, int lepParam, int metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
    TtSemiKinFitter(Parametrization jetParam, Parametrization lepParam, Parametrization metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
    ~TtSemiKinFitter();

    TtSemiEvtSolution addKinFitInfo(TtSemiEvtSolution * asol);

  private:

    void setupFitter();

  private:

    // the kinematic fitter
    TKinFitter * theFitter_;
    // the particles that enter the kinematic fit
    TAbsFitParticle * fitHadb_;
    TAbsFitParticle * fitHadp_;
    TAbsFitParticle * fitHadq_;
    TAbsFitParticle * fitLepb_;
    TAbsFitParticle * fitLepl_;
    TAbsFitParticle * fitLepn_;
    // the constraints on the fit
    TFitConstraintM  * cons1_;
    TFitConstraintM  * cons2_;
    TFitConstraintM  * cons3_;
    TFitConstraintM  * cons4_;
    TFitConstraintM  * cons5_;
    // other parameters
    Parametrization jetParam_, lepParam_, metParam_;
    bool doNeutrinoResol_;
    int maxNrIter_;
    double maxDeltaS_;
    double maxF_;
    std::vector<int> constraints_;
    std::vector<double> translateCovM(TMatrixD &);

};


#endif
