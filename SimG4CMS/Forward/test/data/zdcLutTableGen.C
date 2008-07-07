static const int lutIndxMax1 = 4294967295;  //32 bit integer
static const int lutIndxMax2 = 4294967295;  //32 bit integer
static const int binningIndxMax = 9;
static const int maxIndxMax  = 9;
static const int partLutIndxMax = 64;

TH1I* lutPartIDLut = new TH1I("lutPartIDLut","lutPartIDLut",partLutIndxMax,0,partLutIndxMax);
TH1I* binInfo = new TH1I("binInfo","binInfo",binningIndxMax,0,binningIndxMax);
TH1I* binFactor = new TH1I("binFactor","binFactor",maxIndxMax,0,maxIndxMax);
TH1I* binStep = new TH1I("binStep","binStep",maxIndxMax,0,maxIndxMax);
TH1I* maxBitsInfo = new TH1I("maxBitsInfo","maxBitsInfo",maxIndxMax,0,maxIndxMax);
TH2I* lutMatrixEAverage =new TH2I("lutMatrixEAverage","lutMatrixEAverage",lutIndxMax1,0,lutIndxMax1,lutIndxMax2,0,lutIndxMax2);
TH2I* lutMatrixESigma   =new TH2I("lutMatrixESigma","lutMatrixESigma",lutIndxMax1,0,lutIndxMax1,lutIndxMax2,0,lutIndxMax2);
TH2I* lutMatrixEDist    =new TH2I("lutMatrixEDist","lutMatrixEDist",lutIndxMax1,0,lutIndxMax1,lutIndxMax2,0,lutIndxMax2);


int zdcLutTableGen(){
  
  TFile f("zdcLutTable.root","RECREATE");
  fillBinInfo();
  fillMaxBitsInfo();
  fillBinFactor();
  fillBinStep();
  fillParticleTableInfo();
  //fillLut();
  binInfo->Write();
  maxBitsInfo->Write();
  lutPartIDLut->Write();
  lutMatrixEAverage->Write();
  lutMatrixESigma->Write();
  lutMatrixEDist->Write();
  f.Close();
  return 0;
}


int fillLut(){
  double dE =0;
  double dEsigma =0;
  double dEdist =0;
  for(int ienergy= 0; ienergy <= int(maxBitsInfo->GetBinContent(1)*binFactor->GetBinContent(1)); ienergy+=binStep->GetBinContent(1))
    for(int itheta=  -int(maxBitsInfo->GetBinContent(2)*binFactor->GetBinContent(2)); 
	itheta <= int(maxBitsInfo->GetBinContent(2)*binFactor->GetBinContent(2)); itheta+=binStep->GetBinContent(2))
      for(int iphi =  - int(maxBitsInfo->GetBinContent(3)*binFactor->GetBinContent(3)); 
	  iphi <= int(maxBitsInfo->GetBinContent(3)*binFactor->GetBinContent(3)); iphi+=binStep->GetBinContent(3))
	for(int iside= 0; iside <= int(maxBitsInfo->GetBinContent(4)*binFactor->GetBinContent(4)); iside+=binStep->GetBinContent(4))
	  for(int isection= 0; isection <= int(maxBitsInfo->GetBinContent(5)*binFactor->GetBinContent(5)); isection+=binStep->GetBinContent(5))
	    for(int channel = 0; channel <= int(maxBitsInfo->GetBinContent(6)*binFactor->GetBinContent(6)); channel+=binStep->GetBinContent(6))
	      for(int ixin=  - int(maxBitsInfo->GetBinContent(7)*binFactor->GetBinContent(7)); 
		  ixin <= int(maxBitsInfo->GetBinContent(7)*binFactor->GetBinContent(7)); ixin+=binStep->GetBinContent(7))
		for(int iyin=  - int(maxBitsInfo->GetBinContent(8)*binFactor->GetBinContent(8)); 
		    iyin <= int(maxBitsInfo->GetBinContent(8)*binFactor->GetBinContent(8)); iyin+=binStep->GetBinContent(8))
		  for(int izin= - int(maxBitsInfo->GetBinContent(9)*binFactor->GetBinContent(9)); 
		      izin <= int(maxBitsInfo->GetBinContent(9)*binFactor->GetBinContent(9)); izin+=binStep->GetBinContent(9))
		    for(int iparCode= 0; iparCode <= int(maxBitsInfo->GetBinContent(10)*binFactor->GetBinContent(10));iparCode+=binStep->GetBinContent(10)){
		      int partID = lutPartIDLut->GetBinContent(iparCode);
		      std::cout<<ienergy<<" "
			       <<itheta<<" "
			       <<iphi<<" "
			       <<iside<<" "
			       <<isection<<" "
			       <<channel<<" "
			       <<ixin<<" "
			       <<iyin<<" "
			       <<izin<<" "
			       <<iparCode<<" "
			       <<std::endl;
		      dE = 11111.000;
		      dEsigma = 11.00;
		      dEdist = 1.0;
		      long int iLutIndex1 = encode1(iphi,itheta,ixin,iyin,izin);
		      long int iLutIndex2 = encode2(ienergy,isection,iside,channel,iparCode);
		      lutMatrixEAverage->SetBinContent(iLutIndex1,iLutIndex1,dE);
		      lutMatrixESigma->SetBinContent(iLutIndex1,iLutIndex1,dEsigma);
		      lutMatrixEDist->SetBinContent(iLutIndex1,iLutIndex1,dEdist);      
		    }
  return 0;
}

int fillMaxBitsInfo(){
  // maxBitsInfo*binFactor = max value of variable
  maxBitsInfo->SetBinContent(1,512); // energy
  maxBitsInfo->SetBinContent(2,64);  // theta
  maxBitsInfo->SetBinContent(3,64);  // phi
  maxBitsInfo->SetBinContent(4,2);   // detector side 
  maxBitsInfo->SetBinContent(5,4);   // detector section 
  maxBitsInfo->SetBinContent(6,8);   // detector channel
  maxBitsInfo->SetBinContent(7,16);  // X
  maxBitsInfo->SetBinContent(8,16);  // Y
  maxBitsInfo->SetBinContent(9,32);  // Z
  maxBitsInfo->SetBinContent(10,64);  // pid
  return 0;
  }

int fillBinInfo(){
  binInfo->SetBinContent(1,50000); // 20 GeV bins 50,000 -- 100 GeV 10,000 -- 500 GeV 2,000, -- 5000 GeV 200
  binInfo->SetBinContent(2,64);  // 3 degree theta binning 
  binInfo->SetBinContent(3,64);  // 3 degree phi binning 
  binInfo->SetBinContent(4,1);  // side
  binInfo->SetBinContent(5,1);  // section 
  binInfo->SetBinContent(6,1);  // channel
  binInfo->SetBinContent(7,10); // x 1 cm bin
  binInfo->SetBinContent(8,10); // y 1 cm bin
  binInfo->SetBinContent(9,40); // z 4 cm bin
  binInfo->SetBinContent(10,1); // 1 bin part ID
  return 0;
}

int fillBinFactor(){
  binFactor->SetBinContent(1,20); // 20 GeV bins 50000, 100 GeV 10,000 , etc.
  binFactor->SetBinContent(2,2);  // 3 degree theta binning 
  binFactor->SetBinContent(3,3);  // 3 degree phi binning 
  binFactor->SetBinContent(4,1);  // side
  binFactor->SetBinContent(5,1);  // section 
  binFactor->SetBinContent(6,1);  // channel
  binFactor->SetBinContent(7,1); // x 1 cm bin
  binFactor->SetBinContent(8,1); // y 1 cm bin
  binFactor->SetBinContent(9,4); // z 4 cm bin
  binFactor->SetBinContent(10,1); // 1 bin part ID
  return 0;
}

int fillBinStep(){
  binStep->SetBinContent(1,195); // 20 GeV bins, 100 GeV 10,000 , 500, etc.
  binStep->SetBinContent(2,64);  // 3 degree theta binning 
  binStep->SetBinContent(3,64);  // 3 degree phi binning 
  binStep->SetBinContent(4,1);  // side
  binStep->SetBinContent(5,1);  // section 
  binStep->SetBinContent(6,1);  // channel
  binStep->SetBinContent(7,1); // x 1 cm bin
  binStep->SetBinContent(8,1); // y 1 cm bin
  binStep->SetBinContent(9,4); // z 4 cm bin
  binStep->SetBinContent(10,1); // 1 bin part ID
  return 0;
}

int fillParticleTableInfo(){
  //initialize the array with a non existing pid
  // fill by hand according to PGD convension in cmssw:
  // http://cmslxr.fnal.gov/lxr/source/SimGeneral/HepPDTESSource/data/particle.tbl?v=CMSSW_2_0_6
  lutPartIDLut->SetBinContent(1,11);  // e^-
  lutPartIDLut->SetBinContent(2,-11); // e^+
  lutPartIDLut->SetBinContent(3,13);   // mu^-
  lutPartIDLut->SetBinContent(4,-13); //mu^+
  lutPartIDLut->SetBinContent(5,21);  //g
  lutPartIDLut->SetBinContent(6,22); //gamma
  lutPartIDLut->SetBinContent(7,111); //pi^0
  lutPartIDLut->SetBinContent(8,211); //pi^+
  lutPartIDLut->SetBinContent(9,-211); // pi^-
  lutPartIDLut->SetBinContent(10,311); //K^0             
  lutPartIDLut->SetBinContent(11,321); //K^+           
  lutPartIDLut->SetBinContent(12,-321); //K^-          
  lutPartIDLut->SetBinContent(13,2112); //n^0 
  lutPartIDLut->SetBinContent(14,-2112); //n~^0
  lutPartIDLut->SetBinContent(15,2212); //p^+  
  lutPartIDLut->SetBinContent(16,-2212); //p~^-
  lutPartIDLut->SetBinContent(17,1000010020); //Deuterium    
  lutPartIDLut->SetBinContent(18,1000010030); //Tritium      
  lutPartIDLut->SetBinContent(19,1000020030); //He3          
  lutPartIDLut->SetBinContent(20,100002004); //Alpha-(He4)  
  lutPartIDLut->SetBinContent(partLutIndxMax,666);			   
  return 0;
}

void decode1(const unsigned long & lutidx,int& iphi, int& itheta, int& ix,int& iy, int& iz){
  int iphisgn = (lutidx>>29)&1;
  int ithsgn  = (lutidx>>28)&1;
  int izsgn   = (lutidx>>27)&1;
  int iysgn   = (lutidx>>26)&1;
  int ixsgn   = (lutidx>>25)&1;
  itheta = (lutidx>>19)&63;
  iphi = (lutidx>>13)&63;
  iz = (lutidx>>8)&31;
  iy = (lutidx>>4)&15;
  ix = (lutidx)&15;

  if(ithsgn == 0)itheta*= -1;
  if(iphisgn == 0)iphi*= -1;
  if(izsgn == 0)iz*= -1;
  if(iysgn == 0)iy*= -1;
  if(ixsgn == 0)ix*= -1;
  return;
}

void decode2(const unsigned long & lutidx,int& ien, int& isec, int& isid, int& icha, int& iparID){
  ien = (lutidx>>12)&511;
  iparID = (lutidx>>6)&63;
  icha = (lutidx>>3)&7;
  isec = (lutidx>>1)&3;
  isid = 1 +(lutidx&1);
  return;
}

unsigned long encode1(int iphi, int itheta, int ix, int iy, int iz){
  int ixsgn = 1;
  if(ix<0){
    ix = -ix;
    ixsgn = 0;
  }
  int iysgn = 1;
  if(iy<0){
    iy = -iy;
    iysgn = 0;
  }
  int izsgn = 1;  
  if(iz<0){
    iz = -iz;
    izsgn = 0;
  }
  int ithsgn = 1;
  if(itheta<0){
    itheta = -itheta;
    ithsgn = 0;
  }
  int iphsgn = 1;
  if(iphi<0){
    iphi = -iphi;
    iphsgn = 0;
  }

  unsigned long lutindex = (iphsgn&1)<<29;
  lutindex += (ithsgn&1) <<28;
  lutindex += (izsgn&1)  <<27;
  lutindex += (iysgn&1)  <<26;
  lutindex += (ixsgn&1)  <<25;    //bits 25
  lutindex += (itheta&63)<<19;    //bits 19-24
  lutindex += (iphi&63)  <<13;    //bits 13-18
  lutindex += (iz&31)    <<8;     //bits  8-12
  lutindex += (iy&15)    <<4;     //bits  4- 7
  lutindex += (ix&15);            //bits  0- 3

  //  int newiphi, newitheta, newix, newiy, newiz; 
  //  decode1(lutindex, newiphi, newitheta, newix, newiy, newiz);    
  return lutindex;

}

unsigned long encode2(int ien, int isec, int isid, int icha, int iparID){
  unsigned long  lutindex = (ien&511)<<12;   //bits  12-20
  lutindex += (iparID&63)<<6;                //bits  6-11
  lutindex += (icha&7)   <<3;                //bits  3- 5
  lutindex += (isec&3)   <<1;                //bits  1- 2
  lutindex += ((isid-1)&1);                  //bits  0
  //int newien, newisec, newisid, newicha, newipar; 
  //decode2(lutindex, newien, newisec, newisid, newicha, newipar);    
  return lutindex;
}

