{
  TFile f("histos.root");

  TH1I *barrelHist = (TH1I*)f.Get("Barrel_energy");
  TH1I *barrel_fgvb = (TH1I*)f.Get("Barrel_fgvb");
  TH1I *barrel_ttf = (TH1I*)f.Get("Barrel_ttf");
  TH1I *endcapHist = (TH1I*)f.Get("Endcap_energy");
  TH1I *endcap_fgvb = (TH1I*)f.Get("Endcap_fgvb");
  TH1I *endcap_ttf = (TH1I*)f.Get("Endcap_ttf");
  
  std::cout <<std::endl;
  std::cout <<"[OVAL] Mean of Barrel energy histo "<<barrelHist->GetMean()<<std::endl;
  std::cout <<"[OVAL] Mean of barrel fgvb histo "<<barrel_fgvb->GetMean()<<std::endl;
  std::cout <<"[OVAL] Mean of barrel ttf histo " <<barrel_ttf->GetMean()<<std::endl;
  std::cout <<"[OVAL] Mean of Endcap energy histo "<<endcapHist->GetMean()<<std::endl;
  std::cout <<"[OVAL] Mean of Endcap fgvb histo "<<endcap_fgvb->GetMean()<<std::endl;
  std::cout <<"[OVAL] Mean of endcap ttf histo " <<endcap_ttf->GetMean()<<std::endl;
  return;
}
