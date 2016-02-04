
#ifndef PLOT_FUNCTION
#define PLOT_FUNCTION


int Color [] = {1,4,2,8,6,9,3,7,5};
int Marker[] = {23,22,21,20,29,27,2};
int Style [] = {1,2,5,7,9,10};


TObject* GetObjectFromPath(TDirectory* File, string Path, bool GetACopy=false)
{
   size_t pos = Path.find("/");
   if(pos < 256){
      string firstPart = Path.substr(0,pos);
      string endPart   = Path.substr(pos+1,Path.length());
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

void MakeDirectories(string path){
   size_t pos = 0;
   
   while(pos!=string::npos){
      pos = path.find("/",pos+1);
      if(pos!=string::npos){
         system( (string("mkdir ") + path.substr(0,pos)).c_str());
      }
   }
}

void SaveCanvas(TCanvas* c, string path, string name, bool OnlyPPNG=false){
   string tmppath = path;
   if(tmppath[tmppath.length()-1]!='/')tmppath += "_";
   tmppath += name;

   string filepath;
   filepath = tmppath + ".png"; c->SaveAs(filepath.c_str()); if(OnlyPPNG)return;
   filepath = tmppath +  ".eps"; c->SaveAs(filepath.c_str());
   filepath = tmppath + ".C"  ; c->SaveAs(filepath.c_str());
}

//void DrawPreliminary(int Type, double X=0.28, double Y=0.98, double W=0.85, double H=0.95){
//void DrawPreliminary(double Lumi, double X=0.12, double Y=1.00, double W=0.80, double H=0.945){  //USED FOR PAS
//void DrawPreliminary(double Lumi, double X=0.42, double Y=0.98, double W=0.82, double H=0.945){
void DrawPreliminary(double Lumi, double X=0.40, double Y=0.995, double W=0.82, double H=0.945){
   TPaveText* T = new TPaveText(X,Y,W,H, "NDC");
   T->SetFillColor(0);
   T->SetTextAlign(22);
   if(Lumi<0 )T->AddText("CMS Preliminary   #sqrt{s} = 7 TeV");

   if(Lumi>0 ){
      char tmp[2048];
//      sprintf(tmp,"CMS Preliminary 2010 : L_{int} =%4.1f nb^{-1}  at  #sqrt{s} = 7 TeV",Lumi*1000.0);
//      sprintf(tmp,"CMS Preliminary 2010 : %4.1f nb^{-1}    #sqrt{s} = 7 TeV",Lumi*1000.0);
//      sprintf(tmp,"CMS Preliminary 2010   #sqrt{s} = 7TeV   %4.1f nb ^{-1}",Lumi*1000.0);
//      sprintf(tmp,"CMS Preliminary 2010   #sqrt{s} = 7TeV   %3.0f nb ^{-1}",Lumi*1000.0);
//      sprintf(tmp,"CMS Preliminary 2010   #sqrt{s} = 7TeV   %4.2f pb ^{-1}",Lumi*1.0); //USED FOR PAS
//      sprintf(tmp,"CMS 2010   #sqrt{s} = 7 TeV   %4.2f pb ^{-1}",Lumi*1.0);
//      sprintf(tmp,"CMS   #sqrt{s} = 7 TeV   %4.2f pb ^{-1}",Lumi*1.0);
      sprintf(tmp,"CMS Preliminary   #sqrt{s} = 7 TeV   %4.0f pb ^{-1}",Lumi*1.0);

      T->AddText(tmp);
   }
   T->Draw("same");
}

void DrawLegend (TObject** Histos, std::vector<string> legend, string Title, string Style, double X=0.79, double Y=0.92, double W=0.20, double H=0.05)
{
   int    N             = legend.size();
   
   if(legend[0]!=""){
      TLegend* leg;
      leg = new TLegend(X,Y,X-W,Y - N*H);
      leg->SetFillColor(0);
      leg->SetBorderSize(0);
      //leg->SetTextAlign(32);
      if(Title!="")leg->SetHeader(Title.c_str());

      if(Style=="DataMC"){
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
            leg->AddEntry(temp, legend[i].c_str() ,Style.c_str());
         }
      }
      leg->Draw();
   }
} 


void DrawStatBox(TObject** Histos, std::vector<string> legend, bool Mean, double X=0.15, double Y=0.93, double W=0.15, double H=0.03)
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



void DrawTH2D(TH2D** Histos, std::vector<string> legend, string Style, string Xlegend, string Ylegend, double xmin, double xmax, double ymin, double ymax)
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
   Histos[0]->Draw(Style.c_str());
   for(int i=1;i<N;i++){
        sprintf(Buffer,"%s same",Style.c_str());
        Histos[i]->Draw(Buffer);
   }
}


void DrawSuperposedHistos(TH1** Histos, std::vector<string> legend, string Style,  string Xlegend, string Ylegend, double xmin, double xmax, double ymin, double ymax, bool Normalize=false)
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
        if(xmin!=xmax)Histos[i]->SetAxisRange(xmin,xmax,"X");
        if(ymin!=ymax)Histos[i]->SetAxisRange(ymin,ymax,"Y");
        Histos[i]->SetFillColor(0);
        Histos[i]->SetMarkerStyle(Marker[i]);
        Histos[i]->SetMarkerColor(Color[i]);
        Histos[i]->SetMarkerSize(1.5);
        Histos[i]->SetLineColor(Color[i]);
        Histos[i]->SetLineWidth(2);
       if(Style=="DataMC" && i==0){
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
   if(Style=="DataMC"){
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
      Histos[HistoHeighest]->Draw(Style.c_str());
      for(int i=0;i<N;i++){
           if(i==HistoHeighest)continue;
           if(Style!=""){
              sprintf(Buffer,"same %s",Style.c_str());
           }else{
              sprintf(Buffer,"same");
           }
           Histos[i]->Draw(Buffer);
      }
   }
}


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


#endif
