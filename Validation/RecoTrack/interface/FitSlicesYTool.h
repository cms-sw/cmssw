#ifndef FitSlicesYTool_h
#define FitSlicesYTool_h

#include <TH2F.h>
#include <TH1F.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <string>

class FitSlicesYTool {
 public:
  FitSlicesYTool(TH2F*);
  ~FitSlicesYTool();
  void getFittedMean(MonitorElement*);
  void getFittedSigma(MonitorElement*);
  void getFittedMeanWithError(MonitorElement*);
  void getFittedSigmaWithError(MonitorElement*);
 private:
  TH1* h0;
  TH1* h1;
  TH1* h2;
  TH1* h3;
};

#endif
