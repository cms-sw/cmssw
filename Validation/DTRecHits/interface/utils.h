#ifndef utils_h
#define utils_h
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TMath.h"
#include "TStyle.h"
class Tutils {
 public:
  void drawGFit(TH1 * h1, float nsigmas, float min, float max);
  void drawGFit(TH1 * h1, float min, float max);
  void drawGFit(TH1 * h1, float min, float max, float minfit, float maxfit);
  void setStyle(TH1 *histo);
  void setStyle(TH2 *histo);
  void plotAndProfileX (TH2* h2, float min, float max, bool profile=false);
 private:
  TStyle * getStyle(TString name);
  TStyle * mystyle;
  
};
#endif
