// Original Author:  Loic Quertenmont

#ifndef PLOT_FUNCTION
#define PLOT_FUNCTION


int Color [] = {1,4,2,8,6,9,3,7,5};
int Marker[] = {20,22,21,23,29,27,2};
int Style [] = {1,2,5,7,9,10};
int GraphStyle [] = {20, 21, 22, 23, 24, 25};

// handfull function to get one TObject from a complex cirectory stucture in a file
TObject* GetObjectFromPath(TDirectory* File, std::string Path, bool GetACopy=false)
{
   size_t pos = Path.find("/");
   if(pos < 256){
      std::string firstPart = Path.substr(0,pos);
      std::string endPart   = Path.substr(pos+1,Path.length());
      TDirectory* TMP = (TDirectory*)File->Get(firstPart.c_str());
      if(TMP!=NULL)return GetObjectFromPath(TMP,endPart,GetACopy);

      printf("BUG: %s\n",Path.c_str());
      return NULL;
   }else{
      if(GetACopy){
         return (File->Get(Path.c_str()))->Clone();
      }else{
         return File->Get(Path.c_str());
      }
   }
}

// similar to the above code
TObject* GetObjectFromPath(TDirectory* Container, TDirectory* File, std::string Path, bool GetACopy=false){
   TObject* toreturn = GetObjectFromPath(File,Path,GetACopy);
   if(TH1* th1 = dynamic_cast<TH1*>(toreturn))th1->SetDirectory(Container);
   return toreturn;
}

// create a directory/subdirectory on disk
void MakeDirectories(std::string path){
   system( (std::string("mkdir -p ") + path).c_str());
}

// save a TCanvas on disk in a few different format (mind that 2D plots can be huge if saved in eps/C/pdf)
void SaveCanvas(TCanvas* c, std::string path, std::string name, bool OnlyPPNG=false){
   std::string tmppath = path;
   if(tmppath[tmppath.length()-1]!='/')tmppath += "_";
   tmppath += name;

   std::string filepath;
   filepath = tmppath + ".png"; c->SaveAs(filepath.c_str()); if(OnlyPPNG)return;
   filepath = tmppath +  ".eps"; c->SaveAs(filepath.c_str());
   filepath = tmppath + ".C"  ; c->SaveAs(filepath.c_str());
   filepath = tmppath +  ".pdf"; c->SaveAs(filepath.c_str());
}

// function that add the TPaveText on the current canvas with the "CMS Preliminary...." on top of the Histograms
void DrawPreliminary(string Text, double SQRTS_, double Lumi, double X=0.15, double Y=0.995, double W=0.82, double H=0.945){
   TPaveText* T = new TPaveText(X,Y,W,H, "NDC");
   T->SetTextFont(43);  //give the font size in pixel (instead of fraction)
   T->SetTextSize(15);  //font size
   T->SetFillColor(0);
   T->SetTextAlign(22);
   char tmp[2048];
 
   char energy[128];
   if(SQRTS_==78.0 || SQRTS_==87.0){
      sprintf(energy, "#sqrt{s} = %1.0f+%1.0f TeV",7.0,8.0);
   }else if(SQRTS_>0.0){
      sprintf(energy, "#sqrt{s} = %1.0f TeV",SQRTS_);
   }else{
      sprintf(energy, " ");
   }
   
   char LumiText[1024];
   if(Lumi<=0 ){
      sprintf(LumiText,"CMS Preliminary   %s",energy);
   }
   if(Lumi>0 ){
     sprintf(LumiText,"CMS Preliminary   %s   %1.1f fb ^{-1}",energy, Lumi*0.001);
     //   sprintf(tmp,"CMS Preliminary  "); 
   }

   if(Text!=""){
     sprintf(tmp,"%s   -   %s",Text.c_str(), LumiText);      
   }else{
     sprintf(tmp,"%s",LumiText);
   }

   T->AddText(tmp);
   T->Draw("same");
}
void DrawPreliminary(double SQRTS_, double Lumi, double X=0.40, double Y=0.995, double W=0.82, double H=0.945){
   DrawPreliminary("",SQRTS_, Lumi, X,Y,W,H);
}


// handfull function to draw the legend associated to a vector of histogram
void DrawLegend (TObject** Histos, std::vector<std::string> legend, std::string Title, std::string Style_, double X=0.79, double Y=0.92, double W=0.20, double H=0.05)
{
   int    N             = legend.size();
   
   if(legend[0]!=""){
      TLegend* leg;
      leg = new TLegend(X,Y,X-W,Y - N*H);
      leg->SetFillStyle(0);
      leg->SetBorderSize(0);
      leg->SetTextFont(43); //give the font size in pixel (instead of fraction)
      leg->SetTextSize(15); //font size
//      leg->SetTextAlign(11);
      if(Title!="")leg->SetHeader(Title.c_str());

      if(Style_=="DataMC"){
         for(int i=0;i<N;i++){
            TH2D* temp = (TH2D*)Histos[i]->Clone();
            temp->SetMarkerSize(1.3);
            if(i==0){
               leg->AddEntry(temp, legend[i].c_str() ,"P");
            }else{
               leg->AddEntry(temp, legend[i].c_str() ,"L");
            }
         }
      }else{
         for(int i=0;i<N;i++){
            TH2D* temp = (TH2D*)Histos[i]->Clone();
            temp->SetMarkerSize(1.3);
            leg->AddEntry(temp, legend[i].c_str() ,Style_.c_str());
         }
      }
      leg->Draw();
   }
} 

// draw the stat box
void DrawStatBox(TObject** Histos, std::vector<std::string> legend, bool Mean, double X=0.15, double Y=0.93, double W=0.15, double H=0.03)
{  
   int    N             = legend.size();
   char   buffer[255];

   if(Mean)H*=3;
   for(int i=0;i<N;i++){
           TPaveText* stat = new TPaveText(X,Y-(i*H), X+W, Y-(i+1)*H, "NDC");
           TH1* Histo = (TH1*)Histos[i];
           sprintf(buffer,"Entries : %i\n",(int)Histo->GetEntries());
           stat->AddText(buffer);

           if(Mean){
           sprintf(buffer,"Mean    : %6.2f\n",Histo->GetMean());
           stat->AddText(buffer);

           sprintf(buffer,"RMS     : %6.2f\n",Histo->GetRMS());
           stat->AddText(buffer);
           }

           stat->SetFillColor(0);
           stat->SetLineColor(Color[i]);
           stat->SetTextColor(Color[i]);
           stat->SetBorderSize(0);
           stat->SetMargin(0.05);
           stat->SetTextAlign(12);
           stat->Draw();
   }
}

// draw a TH2D histogram
void DrawTH2D(TH2D** Histos, std::vector<std::string> legend, std::string Style_, std::string Xlegend, std::string Ylegend, double xmin, double xmax, double ymin, double ymax)
{
   int    N             = legend.size();
   
   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend.c_str());
        Histos[i]->GetYaxis()->SetTitle(Ylegend.c_str());
        Histos[i]->GetYaxis()->SetTitleOffset(1.60);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(0.3);
   }

   char Buffer[256];
   Histos[0]->Draw(Style_.c_str());
   for(int i=1;i<N;i++){
        sprintf(Buffer,"%s same",Style_.c_str());
        Histos[i]->Draw(Buffer);
   }
}

// Draw a list of TH1 and superimposed them
void DrawSuperposedHistos(TH1** Histos, std::vector<std::string> legend, std::string Style_,  std::string Xlegend, std::string Ylegend, double xmin, double xmax, double ymin, double ymax, bool Normalize=false, bool same=false, bool lastBinOverflow=false, bool firstBinOverflow=false)
{
   int    N             = legend.size();

   double HistoMax      = -1;
   int    HistoHeighest = -1;

   for(int i=0;i<N;i++){
        if(!Histos[i])continue;
        if(Normalize && Histos[i]->Integral()!=0)Histos[i]->Scale(1.0/Histos[i]->Integral());
        Histos[i]->SetTitle("");
        Histos[i]->SetStats(kFALSE);
        Histos[i]->GetXaxis()->SetTitle(Xlegend.c_str());
        Histos[i]->GetYaxis()->SetTitle(Ylegend.c_str());
        Histos[i]->GetXaxis()->SetTitleOffset(1.1);
        Histos[i]->GetYaxis()->SetTitleOffset(1.70);
        Histos[i]->GetXaxis()->SetNdivisions(505);
        Histos[i]->GetYaxis()->SetNdivisions(505);
	Histos[i]->GetXaxis()->SetTitleSize(0.05);
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetFillColor(0);
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(1.5);
        Histos[i]->SetLineColor(Color[i]);
        Histos[i]->SetLineWidth(2);
        if(lastBinOverflow) {
          if(xmin!=xmax) {
            int lastBin=Histos[i]->GetXaxis()->FindBin(xmax);
            double sum=0;
            double error=0;
            for(int b=lastBin; b<Histos[i]->GetNbinsX()+2; b++) {sum+=Histos[i]->GetBinContent(b); error+=Histos[i]->GetBinError(b)*Histos[i]->GetBinError(b);}
            Histos[i]->SetBinContent(lastBin, sum);
            Histos[i]->SetBinError(lastBin, sqrt(error));
          }
          else {
            Histos[i]->SetBinContent(Histos[i]->GetNbinsX(), Histos[i]->GetBinContent(Histos[i]->GetNbinsX())+Histos[i]->GetBinContent(Histos[i]->GetNbinsX()+1));
            double error=sqrt(pow(Histos[i]->GetBinError(Histos[i]->GetNbinsX()),2)+pow(Histos[i]->GetBinError(Histos[i]->GetNbinsX()+1),2));
            Histos[i]->SetBinError(Histos[i]->GetNbinsX(), error);
          }
        }
        if(firstBinOverflow) {
          if(xmin!=xmax) {
            int firstBin=Histos[i]->GetXaxis()->FindBin(xmin);
            double sum=0;
            double error=0;
            for(int b=0; b<firstBin; b++) {sum+=Histos[i]->GetBinContent(b); error+=Histos[i]->GetBinError(b)*Histos[i]->GetBinError(b);}
            Histos[i]->SetBinContent(firstBin, sum);
            Histos[i]->SetBinError(firstBin, sqrt(error));
          }
          else {
            Histos[i]->SetBinContent(1, Histos[i]->GetBinContent(1)+Histos[i]->GetBinContent(0));
            double error=sqrt(pow(Histos[i]->GetBinError(1),2)+pow(Histos[i]->GetBinError(0),2));
            Histos[i]->SetBinError(1, error);
          }
	}
       if(Style_=="DataMC" && i==0){
           Histos[i]->SetFillColor(0);
           Histos[i]->SetMarkerStyle(20);
           Histos[i]->SetMarkerColor(1);
           Histos[i]->SetMarkerSize(1);
           Histos[i]->SetLineColor(1);
           Histos[i]->SetLineWidth(2);
       }

        if(Histos[i]->GetMaximum() >= HistoMax){
           HistoMax      = Histos[i]->GetMaximum();
           HistoHeighest = i;
        }
   }

   char Buffer[256];
   if(Style_=="DataMC"){
      if(HistoHeighest==0){
         Histos[HistoHeighest]->Draw("E1");
      }else{
         Histos[HistoHeighest]->Draw("HIST");
      }
      for(int i=0;i<N;i++){
           if(i==HistoHeighest)continue;
           if(i==0){
              Histos[i]->Draw("same E1");
           }else{
              Histos[i]->Draw("same");
           }
      }
   }else{
     if(same) {sprintf(Buffer,"same %s",Style_.c_str());
       Histos[HistoHeighest]->Draw(Buffer);}
     else Histos[HistoHeighest]->Draw(Style_.c_str());
      for(int i=0;i<N;i++){
           if(i==HistoHeighest)continue;
           if(Style_!=""){
	     sprintf(Buffer,"same %s",Style_.c_str());
           }else{
              sprintf(Buffer,"same");
           }
           Histos[i]->Draw(Buffer);
      }
   }
}

// automatically determined what is the best axis ranges for a TH2D
void Smart_SetAxisRange(TH2D* histo){
   double Min=1E50;
   double Max=1E-50;
   for(int x=1;x<=histo->GetNbinsX();x++){
   for(int y=1;y<=histo->GetNbinsY();y++){
      double c = histo->GetBinContent(x,y);
      if(c<Min && c>0)Min=c;
      if(c>Max)Max=c;
   }}   
   if(Max/Min<10 ){Max*= 5.0; Min/= 5.0;}
   else if(Max/Min<100){Max*=10.0; Min/=10.0;}
   histo->SetAxisRange(Min,Max,"Z");
}

// return a TCUTG corresponding to the uncertainty on a xsection
TCutG* GetErrorBand(string name, int N, double* Mass, double* Low, double* High, double MinLow, double MaxHigh){
   TCutG* cutg = new TCutG(name.c_str(),2*N);
   cutg->SetFillColor(kGreen-7);
   for(int i=0;i<N;i++){
      double Min = std::max(Low[i],MinLow);
      cutg->SetPoint( i,Mass[i], Min);
   }
   for(int i=0;i<N;i++){
      double Max = std::min(High[N-1-i],MaxHigh);
      cutg->SetPoint(N+i,Mass[N-1-i], Max);
   }
   return cutg;
}



#endif
