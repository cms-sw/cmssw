// Compare the histograms from the ECAL TPG validation 
void displayHistos( char *Current, char *Reference=0 ){

 TText* te = new TText();
 te->SetTextSize(0.1);
 
 TFile * curfile = new TFile( TString(Current)+".root" );

 //1-Dimension Histogram
 TList* list = curfile->GetListOfKeys();
 TObject*  object = list->First();
 int iHisto = 0; char title[50];
 while (object) {
   iHisto++;
   
   // find histo objects
   std::cout << " Histo number = " << iHisto << " object :" << object->GetName() << std::endl;
   TH1I * h1 = dynamic_cast<TH1I*>( curfile->Get(object->GetName()));

     char titleHisto[50]; 
     char title[50];
     // draw and  compare
     std::cout << " Start draw and save the histograms" << std::endl;
     TCanvas c1;
     c1.SetFillColor(0);
     c1.SetFrameFillColor(0);
   
     h1->SetLineColor(2);
     h1->SetLineStyle(1);
     h1->SetLineWidth(3);
     h1->SetMarkerColor(3);
     
     if (iHisto == 1 || iHisto == 4) 
     	sprintf(titleHisto,"%s [ADC units]",object->GetName());
     else
        sprintf(titleHisto,"%s",object->GetName());
     h1->GetXaxis()->SetTitle(titleHisto);
     h1->SetTitle("TPG validation for the 326 release");
     //TLegend leg(0.6,0.7,0.8,0.9);
     //leg.AddEntry(h1, "Validation of the 320 release", "l");

     h1->Draw();
     //leg.Draw();
     c1.SetLogy();
     
     
     sprintf(title,"%s%s", object->GetName(),".gif");
     c1.Print(title);
   
    // go to next object
    object = list->After(object);
   }
}
