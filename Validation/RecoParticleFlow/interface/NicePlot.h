#ifndef __Validation_RecoParticleFlow_NicePlot__
#define __Validation_RecoParticleFlow_NicePlot__

#include <TH1.h>
#include <TPad.h>

class Style : public TH1 {
private:
  void AddBinContent(Int_t) override {}
  void AddBinContent(Int_t, Double_t) override {}
  Double_t RetrieveBinContent(Int_t) const override { return 0.; }
  void UpdateBinContent(Int_t, Double_t) override {}
};

class Styles {
public:
  Style *s1;
  Style *s2;
  Style *sg1;
  Style *sback;
  Style *spred;
  Style *spblue;
  Style *sgr1;
  Style *sgr2;

  Styles();

  static void FormatHisto(TH1 *h, const Style *s);

  static void FormatPad(TPad *pad, bool grid = true, bool logx = false, bool logy = false);

  static void SavePlot(const char *name, const char *dir);
};

#endif
