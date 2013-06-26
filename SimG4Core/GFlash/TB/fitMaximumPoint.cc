Double_t Pol2(Double_t *x, Double_t *par)
{
  //polynoom(2)     
   Double_t part0  = par[0]; 
   Double_t part1  = par[1]*x[0];
   Double_t part2  = par[2]*x[0]*x[0];
   Double_t fitval = part0 + part1 + part2;
   return fitval;
}

Double_t Pol2_Special(Double_t *x, Double_t *par)
{
  //polynoom(2)     
   Double_t part0  = par[0]; 
   Double_t part1  = -2.*par[2]*par[1]*x[0];
   Double_t part2  = par[2]*x[0]*x[0];
   Double_t fitval = part0 + part1 + part2;
   return fitval;
}


//====================================================
//====================================================
Float_t Fit_MaximumPoint( TH2F *h2, Int_t Ireturn = 0){  // xxx
//====================================================
//====================================================

  // -------------------------------------------
  // 1) Clean beam profile
  //     o all bins:  only bins >  2  entries
  //     o per x-bin: only bins > 75% of maximum 
  // -------------------------------------------
  // get some caracteristics
  Int_t   x_nbins = h2->GetXaxis()->GetNbins();  
  Float_t x_max   = h2->GetXaxis()->GetXmax();
  Float_t x_min   = h2->GetXaxis()->GetXmin();
  Float_t delta_x = h2->GetXaxis()->GetBinWidth(10);
  Int_t   y_nbins = h2->GetYaxis()->GetNbins();  
  Float_t y_max   = h2->GetYaxis()->GetXmax();
  Float_t y_min   = h2->GetYaxis()->GetXmin();
  Float_t delta_y = h2->GetYaxis()->GetBinWidth(10);
  TH2F *IHBeamProf_clean_1 = new TH2F("IHBeamProf_clean_1", "Max sam versus X (clean) ",  x_nbins, x_min, x_max, y_nbins, y_min, y_max); 
  TH2F *IHBeamProf_clean_2 = new TH2F("IHBeamProf_clean_2", "Max sam versus X (clean) ",  x_nbins, x_min, x_max, y_nbins, y_min, y_max); 

  // [1a] Only keep bins with more than 2 entries
  //      Also remember for each x-bin the maximum bin in y that satisfies this result
  Int_t y_binmax_high[500] = {1}; 
  for (Int_t ix=1;ix<x_nbins+1;ix++){         
    for (Int_t iy=1;iy<y_nbins+1;iy++){
       Int_t Nevts_clean = 0;
       if( h2->GetBinContent(ix,iy) > 1 ) {
           IHBeamProf_clean_1->SetBinContent(ix,iy, h2->GetBinContent(ix,iy));
           if( iy > y_binmax_high[ix] ){  y_binmax_high[ix] = iy; }         
       }                 
    }
  }
  // [1b] Only keep events with more than 85% of the maximum
  Int_t y_binmax_low[500] = {1}; 
  for (Int_t ix=1;ix<x_nbins+1;ix++){  
    y_binmax_low[ix]  = (Int_t)( 0.85*(Float_t)( y_binmax_high[ix]));
      for (Int_t iy=y_binmax_low[ix];iy<y_binmax_high[ix]+1;iy++){
        IHBeamProf_clean_2->SetBinContent(ix,iy,IHBeamProf_clean_1->GetBinContent(ix,iy));
      }
  }


  // -----------------------------------------------------------------------
  // 2) Find region to fit
  //     o Make profile by compuing means in every bin (error = RMS/sqrt(N))
  //     o Find maximum
  // -----------------------------------------------------------------------
  TH1F *IHBeamProf_clean_3 = new TH1F((h2->GetName()+(TString("_clprof"))).Data(), "Meam Energy versus X (clean) ",  x_nbins, x_min, x_max);  

  // [2a] Make TH1F containing the mean values
  TH1F *h_temp = new TH1F("h_temp"," Energy distribution for single bin in X", y_nbins, y_min, y_max);
  for (Int_t ix=1;ix<x_nbins+1;ix++){          
      double Nevt_slice = 0;
      //Int_t Nevt_slice = 0;
      for (Int_t iy=1;iy<y_nbins+1;iy++){            
        Int_t Nevt_tmp = (int)IHBeamProf_clean_2->GetBinContent(ix,iy); 
        Nevt_slice += Nevt_tmp;
        h_temp->SetBinContent(iy,Nevt_tmp);        
      }
      Float_t Y_mean       = h_temp->GetMean();
      Float_t Y_rms        = h_temp->GetRMS();
      Float_t Y_mean_error = (Nevt_slice>0) ? Y_rms/sqrt(Nevt_slice) : 9999.;
      IHBeamProf_clean_3->SetBinContent(ix,Y_mean);
      IHBeamProf_clean_3->SetBinError(ix,Y_mean_error);
      printf("%d %f %f %d %f\n",ix,Y_mean,Y_rms,Nevt_slice,Y_mean_error);
  }

  // [2b] Find maximum
  Float_t Y_tmp_max       = -999.;
  Float_t x_max_guess     = 0.;
  Int_t   x_max_guess_bin = 0;
  for (Int_t ix=1;ix<x_nbins+1;ix++){  
    Float_t X_tmp       = IHBeamProf_clean_3->GetBinCenter(ix);
    Float_t Y_tmp       = IHBeamProf_clean_3->GetBinContent(ix);
    Float_t Y_tmp_error = IHBeamProf_clean_3->GetBinError(ix);
    if( (fabs(X_tmp) < 10.) && (Y_tmp > Y_tmp_max) && (Y_tmp_error < 100.)){
      Y_tmp_max       = Y_tmp;
      x_max_guess     = X_tmp; 
      x_max_guess_bin = ix; 
      printf("Xtmp = %5.3f  Ytmp = %5.3f  Ytmp_max = %5.3f\n",X_tmp,Y_tmp,Y_tmp_max);
    }
  }
  printf("Fit: o Start for maximum = %5.3f\n",x_max_guess);
   
  // [2c] Define the fit range  (0.975% and error should be less than 8\%)
  Int_t fitbinmin =  9999;
  Int_t fitbinmax = -9999;
  Float_t x_tmp,y_tmp,e_tmp,er_tmp;
  for (Int_t ix = x_max_guess_bin; ix<x_nbins+1;ix++){ 
       fitbinmax = ix;
       x_tmp     = IHBeamProf_clean_3->GetBinCenter(ix);
       y_tmp     = IHBeamProf_clean_3->GetBinContent(ix);
       e_tmp     = IHBeamProf_clean_3->GetBinError(ix);
       er_tmp    = (y_tmp>0.) ? (e_tmp/y_tmp) : 1.00;
       printf("%3d %f %f %f  --   %f %f\n",ix,x_tmp,y_tmp,e_tmp,y_tmp/Y_tmp_max,er_tmp);
       if( y_tmp < 0.975*Y_tmp_max  && er_tmp < 0.008) break;         
  }  
  for (Int_t ix = x_max_guess_bin; ix>1        ;ix--){ 
       fitbinmin = ix;
       x_tmp     = IHBeamProf_clean_3->GetBinCenter(ix);
       y_tmp     = IHBeamProf_clean_3->GetBinContent(ix);
       e_tmp     = IHBeamProf_clean_3->GetBinError(ix);
       er_tmp    = (y_tmp>0.) ? (e_tmp/y_tmp) : 1.00;
       printf("%3d %f %f %f  --   %f %f\n",ix,x_tmp,y_tmp,e_tmp,y_tmp/Y_tmp_max,er_tmp);
       if( y_tmp < 0.975* Y_tmp_max  && er_tmp < 0.008) break;         
  }

  Double_t posmin_fit = x_min+fitbinmin*delta_x;
  Double_t posmax_fit = x_min+fitbinmax*delta_x;
  printf("     o Fit range = %5.3f -- %5.3f -- %5.3f\n",posmin_fit,x_max_guess,posmax_fit);
  if( fabs(posmax_fit - posmin_fit ) < 4. ){
    printf("Something is wrong with this range: returning dummy value\n");
    posmin_fit = x_max_guess - 6.0;
    posmax_fit = x_max_guess + 6.0;
    return -99.00;
  } 

  // -------------------------------------------------
  // 3) Do the actual fit
  //    o make clone of histogram
  // -------------------------------------------------
  TH1F *h1  = (TH1F *) IHBeamProf_clean_3->Clone();
  // 3a] Do the actual fit
  Double_t fitresults[3]          = {0.};
  Double_t fitresults_errors[3]   = {0.};
  TF1 *f1 = new TF1("f1",Pol2,-50.,50.,3);  
  f1->SetParameters( 1.,1.,1.);
  h1->Fit("f1","Q","",posmin_fit, posmax_fit);
  for(int i=0 ; i< 3 ; i++) { 
    fitresults[i]        = f1->GetParameter(i); 
    fitresults_errors[i] = f1->GetParError(i); 
  }    
  Float_t chi2 = f1->GetChisquare()/f1->GetNDF();  
  Float_t a = fitresults[2]; Float_t da = fitresults_errors[2];
  Float_t b = fitresults[1]; Float_t db = fitresults_errors[1];
  Float_t c = fitresults[0]; Float_t dc = fitresults_errors[0];
  Float_t x0 = (-1.*b)/(2*a);
  printf("  a = %7.2f   b = %7.2f   c = %7.2f\n",a,b,c);
  printf(" da = %7.2f  db = %7.2f  dc = %7.2f\n",da,db,dc);

  cout << "risultati del fit polinomiale: " << fitresults[0] << " " << fitresults[1] << " " << fitresults[2] << endl;

  char myProfTitle[200];
  sprintf(myProfTitle, h2->GetTitle());
  strcat(myProfTitle,"_prof");
  h1->Write(myProfTitle);  
  //h1->Write(); //write the profile
  // ----------------------------
  // 4) Compute uncertainty on x0
  // ----------------------------
  // [4a] compute dxo using the covariance matrix
  // covariance matrix
  Double_t CovMat[3][3];
  gMinuit->mnemat(&CovMat[0][0],3);
  Float_t v11 = CovMat[0][0];  Float_t v12 = CovMat[0][1]; Float_t v13 = CovMat[0][2];
  Float_t v21 = CovMat[1][0];  Float_t v22 = CovMat[1][1]; Float_t v23 = CovMat[1][2];
  Float_t v31 = CovMat[2][0];  Float_t v32 = CovMat[2][1]; Float_t v33 = CovMat[2][2];
  printf("Covariance Matrix:   v11 = %f     v12 = %f  v13 = %f\n",CovMat[0][0],CovMat[0][1],CovMat[0][2]);
  printf("                     v21 = %f     v22 = %f  v23 = %f\n",CovMat[1][0],CovMat[1][1],CovMat[1][2]);
  printf("                     v31 = %f     v32 = %f  v33 = %f\n",CovMat[2][0],CovMat[2][1],CovMat[2][2]);
  // jacobiaan
  Float_t j1  =  b/(2*a*a);
  Float_t j2  = -1./(2*a);
  // determinant covariance martix
  Float_t det =   v11*(v33*v22-v32*v23)-v21*(v33*v12-v32*v13)+v31*(v23*v12-v22*v13);
  printf("Determinant = %f\n",det);
  // inverse matrix
  Float_t v11_i =   v33*v22-v32*v23;   Float_t v12_i = -(v33*v12-v32*v13); Float_t v13_i =   v23*v12-v22*v13;
  Float_t v21_i = -(v33*v21-v31*v23);  Float_t v22_i =   v33*v11-v31*v13 ; Float_t v23_i = -(v23*v11-v21*v13) ;
  Float_t v31_i =   v32*v21-v31*v22;   Float_t v32_i = -(v32*v11-v31*v12); Float_t v33_i =   v22*v11-v21*v12;
  // variance
  Float_t var     = j1*(j1*v11_i+j2*v12_i)+j2*(j1*v21_i+j2*v22_i);
           var    /= det;
  Float_t dx0     = sqrt(var);    
  printf("Type 1 fit:  o  x0 = %f  +/- %f\n",x0,dx0);


  // ---------------------
  // 5) Second type of fit
  // ---------------------  
  // 5a] Do the actual fit
  TH1F *h0  = (TH1F *) IHBeamProf_clean_3->Clone();
  Double_t fitresults0[3]          = {0.};
  Double_t fitresults0_errors[3]   = {0.};
  TF1 *f0 = new TF1("f0",Pol2_Special,-50.,50.,3);  
  f0->SetParameters( -1.,x_max_guess,5000.);
  h0->Fit("f0","Q","",posmin_fit, posmax_fit);
  for(int i=0 ; i< 3 ; i++) { 
    fitresults0[i]        = f0->GetParameter(i); 
    fitresults0_errors[i] = f0->GetParError(i); 
  }    
  Float_t a0 = fitresults0[2]; Float_t da0 = fitresults0_errors[2];
  Float_t b0 = fitresults0[1]; Float_t db0 = fitresults0_errors[1];
  Float_t c0 = fitresults0[0]; Float_t dc0 = fitresults0_errors[0];

  Double_t CovMat0[3][3];
  gMinuit->mnemat(&CovMat0[0][0],3);
  //printf("Cov0[1][1] = %f\n",CovMat0[1][1]);
  Float_t db0cov = sqrt(CovMat0[1][1]);
  printf("Type 2 fit:  o  x0 = %f  +/- %f %f \n",b0,db0,db0cov);
  delete  IHBeamProf_clean_1;
  delete  IHBeamProf_clean_2;
  delete  IHBeamProf_clean_3;
  delete  h_temp;
  delete  f1;
  delete  f0;

  if(Ireturn == 10) {return x0;}
  if(Ireturn == 11) {return dx0;}
  if(Ireturn == 20) {return b0;}
  if(Ireturn == 21) {return db0cov;}
  if(Ireturn == 30) {return chi2;}

  if(Ireturn == 99) {return c0;}



}
