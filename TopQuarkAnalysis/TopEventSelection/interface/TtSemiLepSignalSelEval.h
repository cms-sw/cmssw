#ifndef TtSemiLepSignalSelEval_h
#define TtSemiLepSignalSelEval_h

#include "Math/VectorUtil.h"
#include "TMath.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLepSignalSel.h"

inline double evaluateTtSemiLepSignalSel(PhysicsTools::MVAComputerCache& mvaComputer,
                                         const TtSemiLepSignalSel& sigsel,
                                         float weight = 1.,
                                         const bool isSignal = false) {
  std::vector<PhysicsTools::Variable::Value> values;

  values.emplace_back("sumEt", sigsel.sumEt());
  values.emplace_back("relEt1", sigsel.Et1());
  values.emplace_back("Abs_lepeta", sigsel.lepeta());
  values.emplace_back("MET", sigsel.MET());

  values.emplace_back("dphiMETlepton", sigsel.dphiMETlepton());

  values.emplace_back("detajet2jet3", sigsel.detajet2jet3());
  values.emplace_back("detajet3jet4", sigsel.detajet3jet4());

  values.emplace_back("mindijetmass", sigsel.mindijetmass());
  values.emplace_back("maxdijetmass", sigsel.maxdijetmass());

  values.emplace_back("mindRjetlepton", sigsel.mindRjetlepton());

  return mvaComputer->eval(values);
}

#endif
