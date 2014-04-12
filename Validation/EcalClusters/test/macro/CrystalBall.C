Double_t CrystalBall(Double_t* x, Double_t* par){
  double norm  = par[0];
  double alpha = par[1];
  double n     = par[2];
  double m0    = par[3];
  double sigma = par[4];

  return norm*evaluate(x[0], m0, sigma, alpha, (int)n);
}

Double_t evaluate(double m, double m0, double sigma, double alpha, int n) {
  n = 5;
  Double_t t = (m - m0)/sigma;
  if ( alpha < 0 ) t = -t;

  Double_t absAlpha = fabs((Double_t)alpha);
  if ( t >= - absAlpha ) {
    return exp(-0.5*t*t);
  }
  else {
    Double_t a = TMath::Power(n/absAlpha,n)*exp(-0.5*absAlpha*absAlpha);
    Double_t b = n/absAlpha - absAlpha;

    return a/TMath::Power(b - t, n);
  }
}


