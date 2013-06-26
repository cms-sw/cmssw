void DoCompare(char *Current, char *Reference=0) {

  static const int NPages = 6;

  TText* te = new TText();
  te->SetTextSize(0.1);
  
  gROOT->ProcessLine(".x HistoCompare.C");
  gStyle->SetCanvasColor(0);
  gStyle->SetOptStat(111111);

  HistoCompare * myPV = new HistoCompare();
  TFile * curfile = new TFile( TString(Current)+".root" );
  TFile * reffile = curfile;
  if (Reference) reffile = new TFile(TString(Reference)+".root");

  
  if (Reference) reffile->ls();
  curfile->ls();


  TCanvas* c1 = new TCanvas("c1","c1");
  c1->Divide(3,2); c1->cd(1); 

  TList* list = reffile->GetListOfKeys();  
  TObject*  object = list->First();
  int iHisto = 0; char title[50];
  while (object) {
    TH1F * h1 = dynamic_cast<TH1F*>( reffile->Get(object->GetName()));
    TH1F * h2 = dynamic_cast<TH1F*>( curfile->Get(object->GetName()));
    bool isHisto = (reffile->Get(object->GetName()))->InheritsFrom("TH1F");

    if (isHisto && h1 && h2 && *h1->GetName()== *h2->GetName()) {
      iHisto++;
      c1->cd( (iHisto-1) % NPages + 1);
      h1->SetLineColor(2);
      h1->SetLineStyle(3);
      h1->DrawCopy();
      if (Reference) {
        h2->SetLineColor(4);
        h2->SetLineStyle(5);
        h2->DrawCopy("Same"); 
        myPV->PVCompute(h1,h2, te);
      }
      if ((iHisto % NPages) == 0) {
        sprintf(title,"%s%s%d%s", Current, "_", iHisto/NPages-1, ".eps");
        c1->Print(title);
      }
    }
    if(Reference) delete h1;
    delete h2;
    object = list->After(object);
  }
  if ((iHisto % NPages) != 0) {
    sprintf(title,"%s%s%d%s", Current, "_", iHisto/NPages, ".eps");
    c1->Print(title);
  }

}

