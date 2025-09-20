
// TrackerPhase2HistUtil.h
// -----------------------------------------------------------------------------
// Helper utility for producing efficiency MonitorElements in Phase-2 tracker
// validation & harvesting code.
//
// Author: Brandi Skipworth
// -----------------------------------------------------------------------------

#ifndef Validation_SiTrackerPhase2V_TrackerPhase2HistUtil_h
#define Validation_SiTrackerPhase2V_TrackerPhase2HistUtil_h

#include <cmath>

#include "DQMServices/Core/interface/DQMStore.h"
#include "TH1.h"

namespace phase2tkutil {
  inline void makeEfficiencyME(TH1* numerator,
                               TH1* denominator,
                               dqm::legacy::MonitorElement* me,
                               const std::string& xAxisTitle) {
    TH1* efficiency = me->getTH1();
    efficiency->Divide(numerator, denominator, 1., 1., "B");
    efficiency->SetMinimum(0.0);
    efficiency->SetMaximum(1.1);
    efficiency->SetStats(false);
    efficiency->GetXaxis()->SetTitle(xAxisTitle.c_str());
    efficiency->GetYaxis()->SetTitle("Efficiency");
  }

}  // namespace phase2tkutil
#endif
