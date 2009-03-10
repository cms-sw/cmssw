
class Style : public TH1F {
};

Style *s1;
Style *s2;
Style *sg1;

void InitNicePlot() {
  s1 = new Style(); 

  s1->SetLineWidth(2);   
  s1->SetLineColor(1);   

  s2 = new Style(); 

  s2->SetLineWidth(2);   
  s2->SetLineColor(4);   

  sg1 = new Style();

  sg1->SetMarkerColor(4);
  sg1->SetLineColor(4);
  sg1->SetLineWidth(2);  
  sg1->SetMarkerStyle(21);

}


void FormatHisto( TH1* h, const Style* s ) {
  h->SetStats(0);
  h->GetYaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetTitleOffset(1.2);
  h->GetXaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetLabelSize(0.045);
  h->GetXaxis()->SetLabelSize(0.045);

  h->SetLineWidth( s->GetLineWidth() );
  h->SetLineColor( s->GetLineColor() );
}

void FormatPad( TPad* pad, bool grid = true) {
  
  
  pad->SetGridx(grid);
  pad->SetGridy(grid);
  
  pad->SetBottomMargin(0.14);
  pad->SetLeftMargin(0.15);
  pad->SetRightMargin(0.05);
  pad->Modified();
  pad->Update();
}
