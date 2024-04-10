#include "SimFastTiming/FastTimingCommon/interface/BTLPulseShape.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <string>

#include "TROOT.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TF1.h"

int main() {
  edm::MessageDrop::instance()->debugEnabled = false;

  const unsigned int histsiz(BTLPulseShape::k1NSecBinsTotal);

  // shape constants and input amplitude

  const double ReferencePulseNpe_ = 100.;
  const double TimeThreshold1_ = 20.;
  const double TimeThreshold2_ = 50.;
  const double Npe_to_V_ = 0.0064;

  const BTLPulseShape theShape;

  const size_t nampli(5);
  const std::array<float, nampli> npe{{8000., 4000., 3500., 1000., 100.}};
  std::vector<TH1F*> histVect;

  // standard display of the implemented shape function
  const int csize = 500;
  TCanvas* showShape = new TCanvas("showShape", "showShape", csize, 2 * csize);

  for (size_t index = 0; index < nampli; index++) {
    const double scale = npe[index] / ReferencePulseNpe_;
    const std::array<float, 3> tATt(
        theShape.timeAtThr(scale, TimeThreshold1_ * Npe_to_V_, TimeThreshold2_ * Npe_to_V_));

    TString name = "BTLShape_" + std::to_string(index);
    histVect.emplace_back(new TH1F(name, "Tabulated BTL shape", histsiz, 0., (float)(histsiz)));

    std::cout << "Tabulated BTL shape, scale vs reference = " << std::fixed << std::setw(6) << std::setprecision(2)
              << scale << " maximum at [" << std::fixed << std::setw(6) << std::setprecision(2) << theShape.indexOfMax()
              << " ] = " << std::fixed << std::setw(6) << std::setprecision(2) << theShape.timeOfMax() << std::endl;
    std::cout << "Time at thresholds:\n"
              << std::fixed << std::setw(8) << std::setprecision(3) << TimeThreshold1_ * Npe_to_V_ << " --> " << tATt[0]
              << "\n"
              << std::fixed << std::setw(8) << std::setprecision(3) << TimeThreshold2_ * Npe_to_V_ << " --> " << tATt[1]
              << "\n"
              << std::fixed << std::setw(8) << std::setprecision(3) << TimeThreshold1_ * Npe_to_V_ << " --> " << tATt[2]
              << "\n"
              << std::endl;

    for (unsigned int i = 0; i <= histsiz; ++i) {
      const double time((i + 0.5) / BTLPulseShape::kNBinsPerNSec);
      const double myShape(theShape(time));
      histVect[index]->SetBinContent(i, myShape * scale);
      histVect[index]->SetBinError(i, 0.001);
      std::cout << " bin = " << std::fixed << std::setw(4) << i << " time (ns) = " << std::fixed << std::setw(6)
                << std::setprecision(3) << time << " shape = " << std::setw(11) << std::setprecision(8)
                << myShape * scale << std::endl;
    }

    showShape->cd();
    histVect[index]->SetStats(kFALSE);
    histVect[index]->Draw("SAME");
  }

  showShape->SaveAs("BTLShape.pdf");

  return 0;
}
