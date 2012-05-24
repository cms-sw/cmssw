{

TColor::InitializeColors(); 
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(40));
color->SetRGB(0.87, 0.73, 0.53); // light brown    
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(41));
color->SetRGB(1.0, 0.1, 0.5); // deep roze    
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(42));
color->SetRGB(0.5, 0.8, 0.1); // light green  
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(43));
color->SetRGB(0.1, 0.5, 0.3); // dark  green  
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(44));
color->SetRGB(0.5, 0.2, 0.8); // blue-violet  
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(45));
color->SetRGB(0.2, 0.6, 0.9); // grey-blue    
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(46));
color->SetRGB(1.0, 0.5, 0.0); // orange-brick 
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(47));
color->SetRGB(0.8, 0.0, 0.0); // brick 
//
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(51));
color->SetRGB(1.0 , 1.0 , 0.8 ); // lightest yellow 
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(52));
color->SetRGB(0.8 , 1.00, 1.00); // lightest blue-cyan       
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(53));
color->SetRGB(1.0 , 0.95, 0.95); // lightest rose
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(54));
color->SetRGB(0.8 , 1.0 , 0.8 ); // lightest green
TColor *color = (TColor*)(gROOT->GetListOfColors()->At(55));
color->SetRGB(1.00, 1.00, 1.00); // white

// gStyle->SetOptStat(0);   

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(432-10);//kCyan-10  //formerly 52
  gStyle->SetTitleSize(0.06, "XYZ");
  gStyle->SetTitleXOffset(0.9);
  gStyle->SetTitleYOffset(1.25);

  gStyle->SetLabelOffset(0.007, "XYZ");
  gStyle->SetLabelSize(0.05, "XYZ");

 
  gStyle->SetTitle("");
  gStyle->SetOptTitle(0);
 
  gStyle->SetHistLineColor(0);//45
  gStyle->SetHistLineStyle(1);
  gStyle->SetHistLineWidth(2);

  gStyle->SetPadColor(0);//52  
  gStyle->SetPadBorderSize(1); 
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadRightMargin(0.15);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(10);//55
}


{
gSystem->Load("libFWCoreFWLite.so");
AutoLibraryLoader::enable();
}
