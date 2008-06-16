// -------------------------------------------
// useful functions
// -------------------------------------------

// return the position [0->49] of the max amplitude crystal
int maxAmplitInMatrix(double myMatrix[])
{
  int maxXtal   = 999;
  double maxADC = -999.;
  
  for (int icry=0; icry<49; icry++) {
    if (myMatrix[icry] > maxADC){
      maxADC  = myMatrix[icry];
      maxXtal = icry;
    }}
  
  return maxXtal;
}

// max amplitude xtal energy - passing the xtal number in the matrix
double ene1x1_xtal(double mymatrix[], int xtal)
{
  double E1x1 = 0.;
  if (mymatrix[xtal]<-50) { E1x1 = -1000.; }
  else { E1x1 = mymatrix[xtal]; }
  return E1x1;
}

// 3x3 matrix energy around the max amplitude xtal 
double ene3x3_xtal(double mymatrix[], int xtal)
{
  double E3x3 = 0.;

  if ( (mymatrix[xtal-8]<-50) || (mymatrix[xtal-7]<-50)  || (mymatrix[xtal-6]<-50)  || (mymatrix[xtal-1]<-50) || (mymatrix[xtal]<-50)  || (mymatrix[xtal+1]<-50)  || (mymatrix[xtal+6]<-50) || (mymatrix[xtal+7]<-50)  || (mymatrix[xtal+8]<-50) ) 
    { E3x3 = -1000.; }
  else 
    { E3x3 = mymatrix[xtal-8] + mymatrix[xtal-7] + mymatrix[xtal-6] + mymatrix[xtal-1] + mymatrix[xtal] + mymatrix[xtal+1] + mymatrix[xtal+6] + mymatrix[xtal+7] + mymatrix[xtal+8]; }

  return E3x3;
}

// 5x5 matrix energy around the max amplitude xtal
double ene5x5_xtal(double mymatrix[], int xtal)
{
  double E5x5 = 0.;

  if( (mymatrix[xtal-16]<-50) || (mymatrix[xtal-15]<-50) || (mymatrix[xtal-14]<-50) || (mymatrix[xtal-13]<-50) || (mymatrix[xtal-12]<-50) || (mymatrix[xtal-9]<-50) || (mymatrix[xtal-8]<-50) || (mymatrix[xtal-7]<-50) || (mymatrix[xtal-6]<-50) || (mymatrix[xtal-5]<-50) || (mymatrix[xtal-2]<-50) || (mymatrix[xtal-1]<-50) || (mymatrix[xtal]<-50) || (mymatrix[xtal+1]<-50) || (mymatrix[xtal+2]<-50) || (mymatrix[xtal+5]<-50) || (mymatrix[xtal+6]<-50) || (mymatrix[xtal+7]<-50) || (mymatrix[xtal+8]<-50) || (mymatrix[xtal+9]<-50) || (mymatrix[xtal+12]<-50) || (mymatrix[xtal+13]<-50) || (mymatrix[xtal+14]<-50) || (mymatrix[xtal+15]<-50) || (mymatrix[xtal+16]<-50) )
    { E5x5 = -1000.; }
  else
    { E5x5 = mymatrix[xtal-16] + mymatrix[xtal-15] + mymatrix[xtal-14] + mymatrix[xtal-13] + mymatrix[xtal-12] + mymatrix[xtal-9] + mymatrix[xtal-8] + mymatrix[xtal-7] + mymatrix[xtal-6] + mymatrix[xtal-5] + mymatrix[xtal-2] + mymatrix[xtal-1] + mymatrix[xtal] + mymatrix[xtal+1] + mymatrix[xtal+2] + mymatrix[xtal+5] + mymatrix[xtal+6] + mymatrix[xtal+7] + mymatrix[xtal+8] + mymatrix[xtal+9] + mymatrix[xtal+12] + mymatrix[xtal+13] + mymatrix[xtal+14] + mymatrix[xtal+15] + mymatrix[xtal+16]; }

  return E5x5;
}

// lateral energy along ieta - syjun
void energy_ieta(double mymatrix[], double *energyieta)
{
  for(int jj = 0 ; jj < 5 ; jj++) energyieta[jj] = 0.0; 

  for(int jj = 0 ; jj < 5 ; jj++){
    for (int ii = 0; ii < 5 ; ii++) energyieta[jj] += mymatrix[(8+jj)+7*ii]; 
  }
}

// lateral energy along iphi - syjun
void energy_iphi(double mymatrix[], double *energyiphi)
{
  for(int jj = 0 ; jj < 5 ; jj++) energyiphi[jj] = 0.0; 

  for(int jj = 0 ; jj < 5 ; jj++){
    for (int ii = 0; ii < 5 ; ii++) energyiphi[jj] += mymatrix[(8+7*jj)+ii]; 
  }
}


// 7x7 matrix energy around the max amplitude xtal
double ene7x7_xtal(double mymatrix[])
{
  double E7x7 = 0.;
  
  for (int ii=0;ii<49;ii++)
    {
      if (mymatrix[ii]<-50)
	E7x7 = -1000.;
      else
	E7x7 += mymatrix[ii];
    }

  return E7x7;
}


// -------------------------------------------
// fitting functions
// -------------------------------------------

// crystal ball fit
double crystalball(double *x, double *par) {
  // par[0]:  mean
  // par[1]:  sigma
  // par[2]:  alpha, crossover point
  // par[3]:  n, length of tail
  // par[4]:  N, normalization
                                
  double cb = 0.0;
  double exponent = 0.0;
  double bla = 0.0;
  
  if (x[0] > par[0] - par[2]*par[1]) {
    exponent = (x[0] - par[0])/par[1];
    cb = exp(-exponent*exponent/2.);
  } else {
    double nenner  = pow(par[3]/par[2], par[3])*exp(-par[2]*par[2]/2.);
    double zaehler = (par[0] - x[0])/par[1] + par[3]/par[2] - par[2];
    zaehler = pow(zaehler, par[3]);
    cb = nenner/zaehler;
  }
  
  if (par[4] > 0.) {
    cb *= par[4];
  }
  return cb;
}

double shapeFunction(double *x, double *par) {

  // par[0] = constant value
  // par[1], par[2], par[3]: a3, a2 x>0 pol3  
  // par[4], par[5], par[6]: a3, a2 x<=0 pol3  
  
  double ret_val;
  if(x[0]>0.)
    ret_val = par[0] + par[1]*x[0]*x[0] + par[2]*x[0]*x[0]*x[0] + par[3]*x[0]*x[0]*x[0]*x[0];
  else
    ret_val = par[0] + par[4]*x[0]*x[0] + par[5]*x[0]*x[0]*x[0] + par[6]*x[0]*x[0]*x[0]*x[0];

  return ret_val;
}
