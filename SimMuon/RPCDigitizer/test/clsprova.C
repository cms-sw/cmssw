void clsprova()
{
  ifstream myStream1("outprova.txt", ios::in);


  TH1F* hclsGen = new TH1F("Cls Gen", "Cls Gen", 30, 0.5,30.5);
  float cls = 0;

  while(!myStream1.eof()){

    myStream1>>cls;
    hclsGen->Fill(cls);

  }
  hclsGen->Draw();

}
