#ifndef FitSlicesYTool_h
#define FitSlicesYTool_h

#include <TH2F.h>
#include <TH1F.h>
#include "DQMServices/Core/interface/MonitorElement.h"
#include <string>

class FitSlicesYTool {
public:
  FitSlicesYTool(){}
  ~FitSlicesYTool(){}
  void run(TH2F*, MonitorElement*,int kind=2);

};

#endif
