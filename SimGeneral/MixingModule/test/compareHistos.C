void compareHistos( char *Current, char *Reference=0 ){

 TText* te = new TText();
 te->SetTextSize(0.1);
 
 TFile * curfile = new TFile( TString(Current)+".root" );
 TFile * reffile = curfile;
 if (Reference) reffile = new TFile(TString(Reference)+".root");


 //1-Dimension Histogram
 TList* list = reffile->GetListOfKeys();  
 TObject*  object = list->First();
 
 int iHisto = 0; char title[50];
 while (object) {
   // find histo objects
   std::cout << " Histo " << object->GetName() << std::endl;
   TH1I * h1 = dynamic_cast<TH1I*>( reffile->Get(object->GetName()));
   TH1I * h2 = dynamic_cast<TH1I*>( curfile->Get(object->GetName()));
   bool isHisto = (reffile->Get(object->GetName()))->InheritsFrom("TH1I");
   
   if (isHisto && h1 && h2 && *h1->GetName()== *h2->GetName()) {
     iHisto++;
     char title[50];
     // draw and  compare
     std::cout << " Start draw and compare" << std::endl;
     TCanvas c1;
     TH1I htemp2;
     h2->Copy(htemp2);// to keep 2 distinct histos

     h1->SetLineColor(2);
     htemp2.SetLineColor(3);
     h1->SetLineStyle(3);
     h1->SetMarkerColor(3);
     htemp2.SetLineStyle(5);
     htemp2.SetMarkerColor(5);
     TLegend leg(0.1, 0.15, 0.2, 0.25);
     leg.AddEntry(h1, "Reference", "l");
     leg.AddEntry(&htemp2, "New ", "l");
   
     h1->Draw();
     htemp2.Draw("same");
    
     leg.Draw();
     sprintf(title,"%s%s", object->GetName(),".gif");
     c1.Print(title);
   }
   
   // go to next object
   object = list->After(object);
   }
}
