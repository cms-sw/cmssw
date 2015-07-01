#include "TauAnalysis/SVfitStandalone/interface/svFitStandaloneAuxFunctions.h"

#include <TMath.h>
#include <Math/VectorUtil.h>

namespace svFitStandalone
{
  //-----------------------------------------------------------------------------
  
  double roundToNdigits(double x, int n)
  {
    double tmp = TMath::Power(10., n);
    if ( x != 0. ) {
      tmp /= TMath::Power(10., TMath::Floor(TMath::Log10(TMath::Abs(x))));
    }
    double x_rounded = TMath::Nint(x*tmp)/tmp;
    //std::cout << "<roundToNdigits>: x = " << x << ", x_rounded = " << x_rounded << std::endl;
    return x_rounded;
  }

  // Adapted for our vector types from TVector3 class
  Vector rotateUz(const ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> >& toRotate, const Vector& newUzVector)
  {
    // NB: newUzVector must be a unit vector !
    Double_t u1 = newUzVector.X();
    Double_t u2 = newUzVector.Y();
    Double_t u3 = newUzVector.Z();
    Double_t up = u1*u1 + u2*u2;

    Double_t fX = toRotate.X();
    Double_t fY = toRotate.Y();
    Double_t fZ = toRotate.Z();

    if ( up > 0. ) {
      up = TMath::Sqrt(up);
      Double_t px = fX;
      Double_t py = fY;
      Double_t pz = fZ;
      fX = (u1*u3*px - u2*py + u1*up*pz)/up;
      fY = (u2*u3*px + u1*py + u2*up*pz)/up;
      fZ = (u3*u3*px -    px + u3*up*pz)/up;
    } else if ( u3 < 0. ) {
      fX = -fX;
      fZ = -fZ;
    } else {}; // phi = 0, theta = pi

    return Vector(fX, fY, fZ);
  }

  LorentzVector boostToCOM(const LorentzVector& comSystem, const LorentzVector& p4ToBoost) {
    Vector boost = comSystem.BoostToCM();
    return ROOT::Math::VectorUtil::boost(p4ToBoost, boost);
  }

  LorentzVector boostToLab(const LorentzVector& rfSystem, const LorentzVector& p4ToBoost) {
    Vector boost = rfSystem.BoostToCM();
    return ROOT::Math::VectorUtil::boost(p4ToBoost, -boost);
  }

  double gjAngleLabFrameFromX(double x, double visMass, double invisMass, double pVis_lab, double enVis_lab, double motherMass, bool& isValidSolution) 
  {
    // CV: the expression for the Gottfried-Jackson angle as function of X = Etau/Evis
    //     was obtained by solving equation (1) of AN-2010/256:
    //       http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2010_256_v2.pdf
    //     for cosThetaGJ
    //    (generalized to the case of non-zero mass of the neutrino system in leptonic tau decays, using Mathematica)

    double x2 = x*x;
    double visMass2 = visMass*visMass;
    double invisMass2 = invisMass*invisMass;
    double pVis2_lab = pVis_lab*pVis_lab;
    double enVis2_lab = enVis_lab*enVis_lab;
    double motherMass2 = motherMass*motherMass;
    double term1 = enVis2_lab - motherMass2*x2;
    double term2 = 2.*TMath::Sqrt(pVis2_lab*enVis2_lab*enVis2_lab*term1);
    double term3 = ((visMass2 - invisMass2) + motherMass2)*pVis_lab*x*TMath::Sqrt(term1);
    double term4 = 2.*pVis2_lab*term1;
    double cosGjAngle_lab1 =  (term2 - term3)/term4;
    double cosGjAngle_lab2 = -(term2 + term3)/term4;
    double gjAngle = 0.;
    if ( TMath::Abs(cosGjAngle_lab1) <= 1. && TMath::Abs(cosGjAngle_lab2) > 1. ) {
      gjAngle = TMath::ACos(cosGjAngle_lab1);
    } else if ( TMath::Abs(cosGjAngle_lab1) > 1. && TMath::Abs(cosGjAngle_lab2) <= 1. ) {
      gjAngle = TMath::ACos(cosGjAngle_lab2);
    } else if ( TMath::Abs(cosGjAngle_lab1) <= 1. && TMath::Abs(cosGjAngle_lab2) <= 1. ) {
      isValidSolution = false;
    } else {
      isValidSolution = false;
    }

    return gjAngle;
  }

  double pVisRestFrame(double visMass, double invisMass, double motherMass)
  {
    double motherMass2 = motherMass*motherMass;
    double pVis = TMath::Sqrt((motherMass2 - square(visMass + invisMass))
                             *(motherMass2 - square(visMass - invisMass)))/(2.*motherMass);
    return pVis;
  }

  Vector motherDirection(const Vector& pVisLabFrame, double angleVisLabFrame, double phiLab) 
  {
    // The direction is defined using polar coordinates in a system where the visible energy
    // defines the Z axis.
    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > motherDirectionVisibleSystem(1.0, angleVisLabFrame, phiLab);

    // Rotate into the LAB coordinate system
    return rotateUz(motherDirectionVisibleSystem, pVisLabFrame.Unit());
  }

  LorentzVector motherP4(const Vector& motherP3_unit, double motherP_lab, double motherEn_lab)
  {
    LorentzVector motherP4_lab = LorentzVector(motherP_lab*motherP3_unit.x(), motherP_lab*motherP3_unit.y(), motherP_lab*motherP3_unit.z(), motherEn_lab);
    return motherP4_lab;
  }

  //-----------------------------------------------------------------------------
  double getMeanOfBinsAboveThreshold(const TH1* histogram, double threshold, int verbosity)
  {
    double mean = 0.;
    double normalization = 0.;
    int numBins = histogram->GetNbinsX();
    for ( int iBin = 1; iBin <= numBins; ++iBin ) {
      double binCenter = histogram->GetBinCenter(iBin);
      double binContent = histogram->GetBinContent(iBin);
      if ( binContent >= threshold ) {
        if ( verbosity ) std::cout << " adding binContent = " << binContent << " @ binCenter = " << binCenter << std::endl;
        mean += (binCenter*binContent);
        normalization += binContent;
      }
    }
    if ( normalization > 0. ) mean /= normalization;
    if ( verbosity ) std::cout << "--> mean = " << mean << std::endl;
    return mean;
  }

  void extractHistogramProperties(const TH1* histogram, const TH1* histogram_density,
                                  double& xMaximum, double& xMaximum_interpol, 
                                  double& xMean,
                                  double& xQuantile016, double& xQuantile050, double& xQuantile084,
                                  double& xMean3sigmaWithinMax, double& xMean5sigmaWithinMax, 
                                  int verbosity)
  {
    // compute median, -1 sigma and +1 sigma limits on reconstructed mass
    if ( verbosity ) std::cout << "<extractHistogramProperties>:" << std::endl;

    if ( histogram->Integral() > 0. ) {
      Double_t q[3];
      Double_t probSum[3];
      probSum[0] = 0.16;
      probSum[1] = 0.50;
      probSum[2] = 0.84;
      (const_cast<TH1*>(histogram))->GetQuantiles(3, q, probSum);
      xQuantile016 = q[0];
      xQuantile050 = q[1];
      xQuantile084 = q[2];
    } else {
      xQuantile016 = 0.;
      xQuantile050 = 0.;
      xQuantile084 = 0.;
    }
    
    xMean = histogram->GetMean();
    
    if ( histogram_density->Integral() > 0. ) {
      int binMaximum = histogram_density->GetMaximumBin();
      xMaximum = histogram_density->GetBinCenter(binMaximum);
      double yMaximum = histogram_density->GetBinContent(binMaximum);
      double yMaximumErr = ( histogram->GetBinContent(binMaximum) > 0. ) ?      
        (yMaximum*histogram->GetBinError(binMaximum)/histogram->GetBinContent(binMaximum)) : 0.;
      if ( verbosity ) std::cout << "yMaximum = " << yMaximum << " +/- " << yMaximumErr << " @ xMaximum = " << xMaximum << std::endl;
      if ( binMaximum > 1 && binMaximum < histogram_density->GetNbinsX() ) {
        int binLeft       = binMaximum - 1;
        double xLeft      = histogram_density->GetBinCenter(binLeft);
        double yLeft      = histogram_density->GetBinContent(binLeft);    
        
        int binRight      = binMaximum + 1;
        double xRight     = histogram_density->GetBinCenter(binRight);
        double yRight     = histogram_density->GetBinContent(binRight); 
        
        double xMinus     = xLeft - xMaximum;
        double yMinus     = yLeft - yMaximum;
        double xPlus      = xRight - xMaximum;
        double yPlus      = yRight - yMaximum;
        
        xMaximum_interpol = xMaximum + 0.5*(yPlus*square(xMinus) - yMinus*square(xPlus))/(yPlus*xMinus - yMinus*xPlus);
      } else {
        xMaximum_interpol = xMaximum;
      }
      if ( verbosity ) std::cout << "computing xMean3sigmaWithinMax:" << std::endl;
      xMean3sigmaWithinMax = getMeanOfBinsAboveThreshold(histogram_density, yMaximum - 3.*yMaximumErr, verbosity);
      if ( verbosity ) std::cout << "computing xMean5sigmaWithinMax:" << std::endl;
      xMean5sigmaWithinMax = getMeanOfBinsAboveThreshold(histogram_density, yMaximum - 5.*yMaximumErr, verbosity);
    } else {
      xMaximum = 0.;
      xMaximum_interpol = 0.;
      xMean3sigmaWithinMax = 0.;
      xMean5sigmaWithinMax = 0.;
    }
  }
  //-----------------------------------------------------------------------------
}
