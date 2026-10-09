// This ROOT macro determines the BTL time-walk corrections fitting the TProfile DeltaTsimvsE
// in the BTL/LocalReco validation with a shifted power-law + constant function:
// f(x) = [0]*(x+[1])^[2] + [3]
//
// The DeltaTsimvsE profile is filled with the UncalibratedRecHit time and therefore it
// still contains the constant bar time offset L/2v. The fit function used here is thus
// f(x) + barTimeOffset, so that the fitted parameters are those of the time-walk
// correction alone.
//
// CMSSW instructions to calculate new time-walk corrections for each aging scenario:
//  - run the step 3 switching off the current time-walk corrections:
//
//       process.mtdUncalibratedRecHits.barrel.timeWalkCorrection = cms.string('0.')
//
//  - run the MTD validation enabling the flag:
//
//       process.btlLocalRecoValid.FillTimeWalkPlots = True
//
// Macro usage:
//   root -l 'fitBTLTimeWalkCorr.C+("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root")'

#include "TFile.h"
#include "TProfile.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TLine.h"
#include "TString.h"
#include <cstdio>

const int minEntriesPerBin = 10;

// --- BTL bar time offset
const double barLength = 5.472;                                        // [cm]
const double lightCollSlope = 0.0915;                                  // [ns/cm]
const double defaultBarTimeOffset = 0.5 * barLength * lightCollSlope;  // [ns]

TProfile* p_deltaT_vs_E = nullptr;
TF1* fitFunc = nullptr;
TGraphErrors* g_residuals = nullptr;

TCanvas* c1 = nullptr;
TCanvas* c2 = nullptr;

void fitBTLTimeWalkCorr(const char* fileName = "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root",
                        const double barTimeOffset = defaultBarTimeOffset) {
  // --- Open the validation ROOT file and get the DeltaTsimvsE profile
  const char* histPath = "DQMData/Run 1/MTD/Run summary/BTL/LocalReco/DeltaTsimvsE";

  TFile* f = TFile::Open(fileName);
  if (!f || f->IsZombie()) {
    printf("Error: could not open file %s\n", fileName);
    return;
  }

  f->GetObject(histPath, p_deltaT_vs_E);

  if (!p_deltaT_vs_E) {
    printf("Error: could not find TProfile 'DeltaTsimvsE' in %s\n", fileName);
    f->Close();
    return;
  }

  // --- Determine the fit range from the first and last bins with non-zero content
  //     and at least minEntriesPerBin entries
  int firstBin = -1, lastBin = -1;
  for (int i = 1; i <= p_deltaT_vs_E->GetNbinsX(); ++i) {
    if (p_deltaT_vs_E->GetBinContent(i) != 0 && p_deltaT_vs_E->GetBinEntries(i) >= minEntriesPerBin) {
      if (firstBin < 0)
        firstBin = i;
      lastBin = i;
    }
  }

  if (firstBin < 0) {
    printf("Error: no bin of 'DeltaTsimvsE' has at least %d entries\n", minEntriesPerBin);
    f->Close();
    return;
  }

  double xMin = p_deltaT_vs_E->GetXaxis()->GetBinLowEdge(firstBin);
  double xMax = p_deltaT_vs_E->GetXaxis()->GetBinUpEdge(lastBin);

  // --- Define the fit function
  fitFunc = new TF1("fitFunc", Form("[0]*pow(x+[1],[2])+[3]+%.10g", barTimeOffset), xMin, xMax);
  fitFunc->SetParameters(1.0, 0.1, -1.0, 0.4 - barTimeOffset);
  fitFunc->SetParNames("Norm", "Shift", "Power", "Const");

  // --- Perform the fit
  p_deltaT_vs_E->Fit(fitFunc, "R");

  // --- Plot the fitted profile
  c1 = new TCanvas("c1", "DeltaTsimvsE fit", 800, 600);
  p_deltaT_vs_E->Draw();
  fitFunc->Draw("same");

  // --- Print the result
  printf("\nBar time offset L/2v subtracted from the fit: %.4g ns\n", barTimeOffset);
  printf("Fit results:\n");
  printf("  Norm  = %.4g +/- %.4g\n", fitFunc->GetParameter(0), fitFunc->GetParError(0));
  printf("  Shift = %.4g +/- %.4g\n", fitFunc->GetParameter(1), fitFunc->GetParError(1));
  printf("  Power = %.4g +/- %.4g\n", fitFunc->GetParameter(2), fitFunc->GetParError(2));
  printf("  Const = %.4g +/- %.4g\n", fitFunc->GetParameter(3), fitFunc->GetParError(3));
  printf("  Chi2/NDF = %.4g / %d = %.4g\n",
         fitFunc->GetChisquare(),
         fitFunc->GetNDF(),
         fitFunc->GetChisquare() / fitFunc->GetNDF());

  printf("\nFunction string, bar time offset excluded (cut & paste into python):\n");
  printf("  %g*pow(x+%g,%g)+%g\n",
         fitFunc->GetParameter(0),
         fitFunc->GetParameter(1),
         fitFunc->GetParameter(2),
         fitFunc->GetParameter(3));

  // Plot the residuals (bin content - fitted function)
  g_residuals = new TGraphErrors();
  int nPoints = 0;
  for (int i = firstBin; i <= lastBin; ++i) {
    double content = p_deltaT_vs_E->GetBinContent(i);
    if (content == 0 || p_deltaT_vs_E->GetBinEntries(i) < minEntriesPerBin)
      continue;
    double x = p_deltaT_vs_E->GetBinCenter(i);
    double residual = content - fitFunc->Eval(x);
    g_residuals->SetPoint(nPoints, x, residual);
    g_residuals->SetPointError(nPoints, p_deltaT_vs_E->GetBinWidth(i) / 2.0, p_deltaT_vs_E->GetBinError(i));
    ++nPoints;
  }
  g_residuals->SetTitle("DeltaTsimvsE fit residuals;E_{RECO} [MeV];Data - Fit");
  g_residuals->SetMarkerStyle(20);

  c2 = new TCanvas("c2", "DeltaTsimvsE fit residuals", 800, 600);
  g_residuals->Draw("AP");
  g_residuals->GetYaxis()->SetTitleOffset(1.3);
  TLine* zeroLine = new TLine(xMin, 0, xMax, 0);
  zeroLine->SetLineStyle(2);
  zeroLine->Draw("same");
}
