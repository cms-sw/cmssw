void compareHistos( char *Current, char *Reference=0 ){

 TText* te = new TText();
 te->SetTextSize(0.1);
 
 TFile * curfile = new TFile( TString(Current)+".root" );
 TFile * reffile = curfile;
 if (Reference) reffile = new TFile(TString(Reference)+".root");


 char * prefix="DQMData/MixingV/Mixing";
 //1-Dimension Histogram
 TDirectory * refDir=reffile->GetDirectory(prefix);
 TDirectory * curDir=curfile->GetDirectory(prefix);
 TList* list = refDir->GetListOfKeys();  
 TObject*  object = list->First();
 int iHisto = 0; char title[50];
 while (object) {
   // find histo objects
   std::cout << " object :" << object->GetName() << std::endl;
   TProfile * h1 = dynamic_cast<TProfile*>( refDir->Get(object->GetName()));
   TProfile * h2 = dynamic_cast<TProfile*>( curDir->Get(object->GetName()));
   bool isHisto = (refDir->Get(object->GetName()))->InheritsFrom("TProfile");
   std::cout << " isHisto = " << isHisto << std::endl;
   if (isHisto && h1 && h2 && *h1->GetName()== *h2->GetName()) {
     iHisto++;
      char title[50];
      // draw and  compare
   std::cout << " Start draw and compare" << std::endl;
   TCanvas c1;
   TProfile htemp2;
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
   htemp2.Draw("Same"); 
   leg.Draw();
   sprintf(title,"%s%s", object->GetName(),".gif");
   c1.Print(title);
   }
   
   // go to next object
   object = list->After(object);
   }
}
