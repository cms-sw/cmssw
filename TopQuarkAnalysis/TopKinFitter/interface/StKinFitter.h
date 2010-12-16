//
// $Id: StKinFitter.h,v 1.3 2010/09/06 11:07:13 snaumann Exp $
//

#ifndef TopKinFitter_StKinFitter_h
#define TopKinFitter_StKinFitter_h

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"

#include "TLorentzVector.h"

#include <vector>

class TKinFitter;
class TAbsFitParticle;
class TFitConstraintM;

class StKinFitter : public TopKinFitter {

  public:

    StKinFitter();
    StKinFitter(int jetParam, int lepParam, int metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
    StKinFitter(Param jetParam, Param lepParam, Param metParam, int maxNrIter, double maxDeltaS, double maxF, std::vector<int> constraints);
    ~StKinFitter();

    StEvtSolution addKinFitInfo(StEvtSolution * asol);

  private:

    void setupFitter();

  private:

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
    Param jetParam_, lepParam_, metParam_;
    std::vector<int> constraints_;
};


#endif
