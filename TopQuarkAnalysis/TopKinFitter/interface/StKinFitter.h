//
// $Id: StKinFitter.h,v 1.5 2013/05/30 20:51:27 gartung Exp $
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
    StKinFitter(int jetParam, int lepParam, int metParam, int maxNrIter, double maxDeltaS, double maxF,const std::vector<int>& constraints);
    StKinFitter(Param jetParam, Param lepParam, Param metParam, int maxNrIter, double maxDeltaS, double maxF, const std::vector<int>& constraints);
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
