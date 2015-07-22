{
cout << "Loading FWLite..." << endl;
gSystem->Load("libFWCoreFWLite");
FWLiteEnabler::enable();

cout << "Redefining colors..." << endl;

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

}
