void MacroTest()
{
  ifstream myStream1("/tmp/trentad/OutBatch/EffGeneralBatch.dat", ios::in);
  ifstream myStream2("/tmp/trentad/OutBatch/NoiseGeneralBatch.dat", ios::in);
  ifstream myStream3("/tmp/trentad/OutBatch/EffRollBatch_637567005.dat", ios::in);
  ifstream myStream4("/tmp/trentad/OutBatch/EffRollBatch_637632541.dat", ios::in);

  TFile* output = new TFile("/tmp/trentad/OutBatch/histoTest_EffNoiseBatch.root","RECREATE");

  //---------------------- General -----------------------------------------

  TH1F* hclsGen = new TH1F("Cls Gen", "Cls Gen", 10, 0.5,10.5);
  TH1F* hBxGen = new TH1F("Bx Gen", "Bx Gen", 15, -7.5,7.5);
  TH1F* hBxGenEff = new TH1F("Bx GenEff", "Bx GenEff", 15, -7.5,7.5);
  TH1F* hProfileGen = new TH1F("Profile Gen", "Profile Gen", 100, 0.5,100.5);
  TH1F* hPosGenCls1 = new TH1F("Pos Gen cls1", "Pos Gen cls1", 20, 0.,1.);
  TH1F* hPosGenCls2 = new TH1F("Pos Gen cls2", "Pos Gen cls2", 20, 0.,1.);
  TH1F* hPosGenCls3 = new TH1F("Pos Gen cls3", "Pos Gen cls3", 20, 0.,1.);

  //---------------------- Roll---------------------------------------------

  TH1F* hclsRoll = new TH1F("Cls Roll", "Cls Roll", 10, 0.5,10.5);
  TH1F* hBxRoll = new TH1F("Bx Roll", "Bx Roll", 15, -7.5,7.5);
  TH1F* hProfileRoll1 = new TH1F("Profile Roll 1", "Profile Roll 1", 100, 0.5,100.5);
  TH1F* hProfileRoll2 = new TH1F("Profile Roll 2", "Profile Roll 2", 100, 0.5,100.5);

  TH1F* hProfileEff = new TH1F("Profile Eff", "Profile Eff", 100, 0.5,100.5);

  TH1F* hPosRollCls1 = new TH1F("Pos Roll cls1", "Pos Roll cls1", 20, 0.,1.);
  TH1F* hPosRollCls2 = new TH1F("Pos Roll cls2", "Pos Roll cls2", 20, 0.,1.);
  TH1F* hPosRollCls3 = new TH1F("Pos Roll cls3", "Pos Roll cls3", 20, 0.,1.);

  //------------------------------------------------------------------------

  int strip= 0 , bx = 0, cls = 0;
  float posX = 0.;
  int counter = 0;

  while(!myStream1.eof()){
       if(counter == 6400000) break;

    myStream1>>strip>>bx>>cls>>posX;
    std::cout<<counter<<"  "<<strip<<"  "<<bx<<"  "<<cls<<"  "<<posX<<std::endl;
    if(strip > 100) continue;
    hclsGen->Fill(cls);
    hBxGen->Fill(bx);
    hBxGenEff->Fill(bx);
    hProfileGen->Fill(strip);

    if(cls == 1) hPosGenCls1->Fill(posX);
    if(cls == 2) hPosGenCls2->Fill(posX);
    if(cls == 3) hPosGenCls3->Fill(posX);
    counter++;
  }

  counter = 0;
  while(!myStream2.eof()){
    if(counter == 6400000) break;
    myStream2>>strip>>bx;

    if(strip > 100) continue;
    hProfileGen->Fill(strip);
    hBxGen->Fill(bx);
    counter++;
  }

  counter = 0;
  while(!myStream3.eof()){
    if(counter == 6400000) break;
    myStream3>>strip>>bx>>cls>>posX;
    if(strip > 100) continue;
    hclsRoll->Fill(cls);
    hBxRoll->Fill(bx);
    hProfileRoll1->Fill(strip);

    if(cls == 1) hPosRollCls1->Fill(posX);
    if(cls == 2) hPosRollCls2->Fill(posX);
    if(cls == 3) hPosRollCls3->Fill(posX);
    counter++;
  }

  counter = 0;
  while(!myStream4.eof()){
    if(counter == 6400000) break;
    myStream4>>strip>>bx>>cls>>posX;
    if(strip > 100) continue;
    hProfileRoll2->Fill(strip);
    //    hBxRoll->Fill(bx);
    counter++;
  }

  for(int i = 1; i <= 100; ++i){
    float raf1 = hProfileRoll2->GetBinContent(i);
    float raf2 = hProfileRoll1->GetBinContent(i);
    if(raf1 > 0){
      float eff = raf2/raf1;
      if(eff<=1){
	float err = sqrt(eff(1-eff)/raf1);
	hProfileEff->SetBinContent(i,eff);
	hProfileEff->SetBinError(i,err);
      }
    }
  }

  output->Write();

}
