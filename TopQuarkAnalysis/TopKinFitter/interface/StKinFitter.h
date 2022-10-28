//
//

#ifndef TopKinFitter_StKinFitter_h
#define TopKinFitter_StKinFitter_h

#include "AnalysisDataFormats/TopObjects/interface/StEvtSolution.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"

#include "TLorentzVector.h"

#include <memory>
#include <vector>

class TKinFitter;
class TAbsFitParticle;
class TFitConstraintM;

class StKinFitter : public TopKinFitter {
public:
  StKinFitter();
  StKinFitter(int jetParam,
              int lepParam,
              int metParam,
              int maxNrIter,
              double maxDeltaS,
              double maxF,
              const std::vector<int>& constraints);
  StKinFitter(Param jetParam,
              Param lepParam,
              Param metParam,
              int maxNrIter,
              double maxDeltaS,
              double maxF,
              const std::vector<int>& constraints);
  ~StKinFitter();

  StEvtSolution addKinFitInfo(StEvtSolution* asol);

private:
  void setupFitter();

private:
  // the particles that enter the kinematic fit
  std::unique_ptr<TAbsFitParticle> fitBottom_;
  std::unique_ptr<TAbsFitParticle> fitLight_;
  std::unique_ptr<TAbsFitParticle> fitLepton_;
  std::unique_ptr<TAbsFitParticle> fitNeutrino_;
  // the constraints on the fit
  std::unique_ptr<TFitConstraintM> cons1_;
  std::unique_ptr<TFitConstraintM> cons2_;
  std::unique_ptr<TFitConstraintM> cons3_;
  // other parameters
  Param jetParam_, lepParam_, metParam_;
  std::vector<int> constraints_;
};

#endif
