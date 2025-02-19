void dumpProfile() {
  
 std::string fileName_ = "Profile_SM10.root";
 TFile *shapeFile_ = TFile::Open(fileName_.c_str(),"old");
 TProfile* PROF_704 = (TProfile*) shapeFile_->Get("SHAPE_XTAL_704");

 ofstream out;
 out.open("dat.txt");

 int nBinsHisto_ = 250;
 std::vector<double> shapeArray(nBinsHisto_,0.0);

 double max = -999;
 int imax = 0;
 for(int ibin=0; ibin < nBinsHisto_; ++ibin) 
   {
     out << "shapeArray[" << ibin << "] = " << PROF_704->GetBinContent(ibin+1) << " ; \n";
     shapeArray[ibin] = PROF_704->GetBinContent(ibin);
     std::cout << "Original shape, ns = " << ibin << " shape = " << shapeArray[ibin] << std::endl;
     if ( shapeArray[ibin] > max ) {
       max = shapeArray[ibin];
       imax = ibin;
     }

   }//loop

 out.close();

 double xMinHisto_ = -1.;
 double xMaxHisto_ = 9.;
 double binw = (xMaxHisto_ - xMinHisto_)/(shapeArray.size());
 int nbins = shapeArray.size()/10;

 float low =  xMinHisto_+(double)(imax-nbins/2+0.5)*binw;
 float up = xMinHisto_+(double)(imax+nbins/2+0.5)*binw;
 
 double* x = new double[nbins];
 double* y = new double[nbins];
 for (int i = 0; i < nbins; i++) {
   x[i] = xMinHisto_ + (double)(imax - nbins/2 + i + 0.5)*binw;
   y[i] = shapeArray[imax - nbins/2 + i];
   std::cout << " x,y = " << x[i] << " " << y[i] << " " << (double)(imax - nbins/2 + i + 0.5) << std::endl;
 }
 TGraph* graph = new TGraph(nbins, x, y);
 graph->Fit("pol3", "V");//"Q 0");
 TF1* fFit = graph->GetFunction("pol3");
 double tMax = fFit->GetMaximumX();

 std:;cout << "Maxiumum = " << tMax << std::endl;

 gStyle->SetOptFit(1111);

 TCanvas *MyC = new TCanvas("MyC","Test canvas",1); 
 MyC->Divide(2,1); 
 MyC->cd(1); 
 PROF_704->Draw(); 
 MyC->cd(2); 
 fFit->Draw(); 
 MyC->SaveAs("PROF_704.jpg");
 
}
