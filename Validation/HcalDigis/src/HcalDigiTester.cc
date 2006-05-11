#include "Validation/HcalDigis/interface/HcalDigiTester.h"


HcalDigiTester::HcalDigiTester(const edm::ParameterSet& iConfig)
{
  // DQM ROOT output
  outputFile_ = iConfig.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " Hcal Digi Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " Hcal Digi Task histograms will NOT be saved";
  }
  
  dbe_ = 0;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();
  
  
  meEtaHE = 0;
  mePhiHE = 0;

 
  meEtaHB = 0;
  mePhiHB = 0;
 
  meEtaHF = 0;
  mePhiHF = 0;

  meEtaHO = 0;
  mePhiHO = 0;

  meDigiSimhitHB = 0;
  meDigiSimhitHF = 0;
  meDigiSimhitHE = 0;
  meDigiSimhitHO = 0;


  meDigiSimhitHBprofile = 0;
  meDigiSimhitHEprofile = 0;
  meDigiSimhitHFprofile = 0;
  meDigiSimhitHOprofile = 0;

  meRatioDigiSimhitHB = 0;
  meRatioDigiSimhitHE = 0;
  meRatioDigiSimhitHF = 0;
  meRatioDigiSimhitHO = 0;

  meSumDigisHB = 0;
  meSumDigisHE = 0;
  meSumDigisHF = 0;
  meSumDigisHO = 0;

  Char_t histo[20];
 
  hcalselector_ = iConfig.getUntrackedParameter<string>("hcalselector", "all");
  subpedvalue_ = iConfig.getUntrackedParameter<bool>("subpedvalue", "true");

 if ( dbe_ ) {
   dbe_->setCurrentFolder("HcalDigiTask");
   
   if (hcalselector_ == "HE" || hcalselector_ == "all" ) {
   sprintf (histo, "HcalDigiTask_Eta_of_digis_HE" ) ;
   meEtaHE = dbe_->book1D(histo, histo, 60 , -3. , 3.);
   sprintf (histo, "HcalDigiTask_Phi_of_digis_HE" ) ;
   mePhiHE = dbe_->book1D(histo, histo, 36 , -3.14159, 3.14159);
   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HE");
   meDigiSimhitHE = dbe_->book2D(histo, histo, 100, 0.,1.,  100, 0., 800.);

   sprintf (histo, "HcalDigiTask_Ratio_energy_digis_vs_simhits_HE");
   meRatioDigiSimhitHE = dbe_->book1D(histo, histo,  200, 0., 1000.);   

   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HE(profile)");
   meDigiSimhitHEprofile = dbe_->bookProfile(histo, histo, 100, 0.,1.,  100, 0., 800.);
   sprintf (histo, "HcalDigiTask_number_of_digis_HE");
   menDigisHE = dbe_->book1D(histo, histo,  40, 0., 200.);  

   sprintf (histo, "HcalDigiTask_sum_over_digis(fC)_HE");
   meSumDigisHE = dbe_->book1D(histo, histo,  100, 0., 800.);  


   }
   if (hcalselector_ == "HB" || hcalselector_ == "all"  ) {
   sprintf (histo, "HcalDigiTask_Eta_of_digis_HB" ) ;
   meEtaHB = dbe_->book1D(histo, histo, 40, -1.74 , 1.74);
   sprintf (histo, "HcalDigiTask_Phi_of_digis_HB" ) ;
   mePhiHB = dbe_->book1D(histo, histo, 72, -3.14159, 3.14159);
   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HB");
   meDigiSimhitHB = dbe_->book2D(histo, histo, 150, 0.,1.5,  100, 0., 800.);

   sprintf (histo, "HcalDigiTask_Ratio_energy_digis_vs_simhits_HB");
   meRatioDigiSimhitHB = dbe_->book1D(histo, histo,  200, 0., 1000.);   

   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HB(profile)");
   meDigiSimhitHBprofile = dbe_->bookProfile(histo, histo, 150, 0.,1.5,  100, 0., 800.); 

   sprintf (histo, "HcalDigiTask_number_of_digis_HB");
   menDigisHB = dbe_->book1D(histo, histo,  40 , 0, 80);  
   
   sprintf (histo, "HcalDigiTask_sum_over_digis(fC)_HB");
   meSumDigisHB = dbe_->book1D(histo, histo,  100, 0., 800.);  

   }

   if (hcalselector_ == "HF" || hcalselector_ == "all"  ) {
   sprintf (histo, "HcalDigiTask_Eta_of_digis_HF" ) ;
   meEtaHF = dbe_->book1D(histo, histo, 100, -5. , 5.);
   sprintf (histo, "HcalDigiTask_Phi_of_digis_HF" ) ;
   mePhiHF = dbe_->book1D(histo, histo, 36, -3.14159, 3.14159);
   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HF");
   meDigiSimhitHF = dbe_->book2D(histo, histo, 70, 0.,70.,  100, 0., 350.);

   sprintf (histo, "HcalDigiTask_Ratio_energy_digis_vs_simhits_HF");
   meRatioDigiSimhitHF = dbe_->book1D(histo, histo,  40, 3., 7.);   

   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HF(profile)");
   meDigiSimhitHFprofile = dbe_->bookProfile(histo, histo, 70, 0.,70.,  100, 0., 350.);
   sprintf (histo, "HcalDigiTask_number_of_digis_HF");
   menDigisHF = dbe_->book1D(histo, histo,  20, 0., 20.);  

  sprintf (histo, "HcalDigiTask_sum_over_digis(fC)_HF");
   meSumDigisHF = dbe_->book1D(histo, histo,  100, 0., 350.);  
   }  

   if (hcalselector_ == "HO" || hcalselector_ == "all"  ) {
   sprintf (histo, "HcalDigiTask_Eta_of_digis_HO" ) ;
   meEtaHO = dbe_->book1D(histo, histo, 40, -1.74 , 1.74);
   sprintf (histo, "HcalDigiTask_Phi_of_digis_HO" ) ;
   mePhiHO = dbe_->book1D(histo, histo, 72, -3.14159, 3.14159);
   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HO");
   meDigiSimhitHO = dbe_->book2D(histo, histo, 100, 0.,0.2,  100, 0., 150.);

   sprintf (histo, "HcalDigiTask_Ratio_energy_digis_vs_simhits_HO");
   meRatioDigiSimhitHO = dbe_->book1D(histo, histo,  140, 0., 1400.);   

   sprintf (histo, "HcalDigiTask_energy_digis_vs_simhits_HO(profile)");
   meDigiSimhitHOprofile = dbe_->bookProfile(histo, histo, 100, 0.,0.2,  100, 0., 150.);
   sprintf (histo, "HcalDigiTask_number_of_digis_HO");
   menDigisHO = dbe_->book1D(histo, histo,  50, 0., 50.);  
   sprintf (histo, "HcalDigiTask_sum_over_digis(fC)_HO");
   meSumDigisHO = dbe_->book1D(histo, histo,  100, 0., 150.);  
  
   }  
    
 }
}
   



HcalDigiTester::~HcalDigiTester()
{
  cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);

}

void HcalDigiTester::endJob() {
 cout << " outputFile_.size() =  " << outputFile_.size() << endl;
  cout << " dbe_ = " << dbe_ << endl; 
 if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}
void HcalDigiTester::beginJob(const edm::EventSetup& c){

}


void
HcalDigiTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  edm::Handle<HBHEDigiCollection> hbhe;
  edm::Handle<HFDigiCollection> hf;
  edm::Handle<HODigiCollection> ho;
  HBHEDigiCollection::const_iterator ihbhe;
  HFDigiCollection::const_iterator ihf;
  HODigiCollection::const_iterator iho;

  edm::ESHandle<CaloGeometry> geometry ;
  iSetup.get<IdealGeometryRecord> ().get (geometry) ;


  // ADC2fC
  edm::ESHandle<HcalDbService> conditions;
  iSetup.get<HcalDbRecord>().get(conditions);
  const HcalQIEShape* shape = conditions->getHcalShape();
  HcalCalibrations calibrations;
  
  CaloSamples tool;

 // loop over the digis
  int ndigis=0;

  //  ******************** BEGIN of Hcal Barrel *****************************
  if (hcalselector_ == "HB"|| hcalselector_ == "all")
    { 
      float fAdcSum = 0;// sum of all ADC counts in terms of fC
      iEvent.getByType (hbhe) ;
      
      for (ihbhe=hbhe->begin();ihbhe!=hbhe->end();ihbhe++)
	{
	  HcalDetId cell(ihbhe->id()); 
	  if (cell.subdet()== 1  ) 
	    {
	      const CaloCellGeometry* cellGeometry =
		geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	      double fEta = cellGeometry->getPosition ().eta () ;
	      double fPhi = cellGeometry->getPosition ().phi () ;
	      
	      if (meEtaHB) meEtaHB->Fill(fEta) ;
	      if (mePhiHB) mePhiHB->Fill(fPhi) ;
	      
	      conditions->makeHcalCalibration(cell, &calibrations);
	      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
	      HcalCoderDb coder (*channelCoder, *shape);
	      coder.adc2fC(*ihbhe,tool);
	      

	      for  (int ii=0;ii<tool.size();ii++)
		{
		  int capid = (*ihbhe)[ii].capid();
		  if (subpedvalue_) fAdcSum+=(tool[ii]-calibrations.pedestal(capid));
		  if (!subpedvalue_) fAdcSum+=(tool[ii]-4.5);
		}
	      ndigis++;
	    }
	}
      if (menDigisHB) menDigisHB->Fill(ndigis);
      ndigis=0;
      edm::Handle<PCaloHitContainer> hcalHits ;
      iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
      
      const PCaloHitContainer * simhitResult = hcalHits.product () ;
      
      float fEnergySimHits = 0; 
      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin () ;
	   simhits != simhitResult->end () ;
	   ++simhits)
	{    
	  HcalDetId detId(simhits->id());
	  //  1 == HB
	  if (detId.subdet()==1  ){  fEnergySimHits += simhits->energy(); }
	}
      if (meDigiSimhitHB) meDigiSimhitHB->Fill( fEnergySimHits, fAdcSum);
      if (meRatioDigiSimhitHB) meRatioDigiSimhitHB->Fill(fAdcSum/fEnergySimHits);
      if (meDigiSimhitHBprofile) meDigiSimhitHBprofile->Fill( fEnergySimHits, fAdcSum);
      if (meSumDigisHB) meSumDigisHB->Fill(fAdcSum);
    }
  
 //  ******************** END of Hcal Barrel *****************************

 //  ******************** BEGIN of Hcal Endcap *****************************
 
  if (hcalselector_ == "HE"|| hcalselector_ == "all"){ 
    float fAdcSum = 0;// sum of all ADC counts in terms of fC
    iEvent.getByType (hbhe) ;

    for (ihbhe=hbhe->begin();ihbhe!=hbhe->end();ihbhe++){
    HcalDetId cell(ihbhe->id()); 
    if (cell.subdet()== 2 ) {
      // && cell.depth() == 1     
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double fEta = cellGeometry->getPosition ().eta () ;
      double fPhi = cellGeometry->getPosition ().phi () ;
      // cout << " PHi = " << fPhi << endl;
      if (meEtaHE) meEtaHE->Fill(fEta) ;
      if (mePhiHE) mePhiHE->Fill(fPhi) ;
      
      conditions->makeHcalCalibration(cell, &calibrations);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*ihbhe,tool);

      for  (int ii=0;ii<tool.size();ii++)
	{
	  int capid = (*ihbhe)[ii].capid();
	  if (subpedvalue_) fAdcSum+=(tool[ii]-calibrations.pedestal(capid));
	  if (!subpedvalue_) fAdcSum+=(tool[ii]-4.5);
	}
      ndigis++;
    }
  }
   if (menDigisHE) menDigisHE->Fill(ndigis);
    ndigis=0;
    edm::Handle<PCaloHitContainer> hcalHits ;
    iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
 
    const PCaloHitContainer * simhitResult = hcalHits.product () ;
 
    float fEnergySimHits = 0; 
    for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin () ;
	 simhits != simhitResult->end () ;
	 ++simhits)
      {    
	HcalDetId detId(simhits->id());
	//  2 == HE
	// && detId.depth()==1
	if (detId.subdet()==2){  fEnergySimHits += simhits->energy(); }
      }
    if (meDigiSimhitHE) meDigiSimhitHE->Fill( fEnergySimHits, fAdcSum);
    if (meRatioDigiSimhitHE) meRatioDigiSimhitHE->Fill(fAdcSum/fEnergySimHits);
    if (meDigiSimhitHEprofile) meDigiSimhitHEprofile->Fill( fEnergySimHits, fAdcSum);
  if (meSumDigisHE) meSumDigisHE->Fill(fAdcSum);
  }

 //  ******************** END of Hcal Endcap *****************************

 //  ******************** BEGIN of Hcal Forward *****************************
 
  if (hcalselector_ == "HF"|| hcalselector_ == "all"){ 
    float fAdcSum = 0;// sum of all ADC counts in terms of fC
    iEvent.getByType (hf) ;
    
    for (ihf=hf->begin();ihf!=hf->end();ihf++){
      HcalDetId cell(ihf->id()); 
      if (cell.subdet()== 4  ) {
	const CaloCellGeometry* cellGeometry =
	  geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
	double fEta = cellGeometry->getPosition ().eta () ;
	double fPhi = cellGeometry->getPosition ().phi () ;
	
	if (meEtaHF) meEtaHF->Fill(fEta) ;
	if (mePhiHF) mePhiHF->Fill(fPhi) ;
	
	conditions->makeHcalCalibration(cell, &calibrations);
	const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
	HcalCoderDb coder (*channelCoder, *shape);
	coder.adc2fC(*ihf,tool);
	
	for  (int ii=0;ii<tool.size();ii++)
	  {
	    int capid = (*ihf)[ii].capid();
	   
	  if (subpedvalue_) fAdcSum+=(tool[ii]-calibrations.pedestal(capid));
	  if (!subpedvalue_) fAdcSum+=(tool[ii]-1.73077);

	  }
	ndigis++;
      }
    }
    if (menDigisHF) menDigisHF->Fill(ndigis);
    ndigis=0;
    edm::Handle<PCaloHitContainer> hcalHits ;
    iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
    
    const PCaloHitContainer * simhitResult = hcalHits.product () ;
    
    float fEnergySimHits = 0; 
    for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin () ;
	 simhits != simhitResult->end () ;
	 ++simhits)
      {    
	HcalDetId detId(simhits->id());
	//  4 == HF
	if (detId.subdet()== 4  ){  fEnergySimHits += simhits->energy(); }
      }

    if (meDigiSimhitHF) meDigiSimhitHF->Fill( fEnergySimHits, fAdcSum);
    if (meRatioDigiSimhitHF) meRatioDigiSimhitHF->Fill(fAdcSum/fEnergySimHits);
    if (meDigiSimhitHFprofile) meDigiSimhitHFprofile ->Fill( fEnergySimHits, fAdcSum);
 if (meSumDigisHF) meSumDigisHF->Fill(fAdcSum);
  }

 //  ******************** END of Hcal Forward *****************************
 //  ******************** BEGIN of Hcal Outer *****************************
 
  if (hcalselector_ == "HO"|| hcalselector_ == "all"){ 
    float fAdcSum = 0;// sum of all ADC counts in terms of fC
    iEvent.getByType (ho) ;
    
    for (iho=ho->begin();iho!=ho->end();iho++){
      HcalDetId cell(iho->id()); 
   
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double fEta = cellGeometry->getPosition ().eta () ;
      double fPhi = cellGeometry->getPosition ().phi () ;
      
      if (meEtaHO) meEtaHO->Fill(fEta) ;
      if (mePhiHO) mePhiHO->Fill(fPhi) ;
      
      conditions->makeHcalCalibration(cell, &calibrations);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*iho,tool);
      
      for  (int ii=0;ii<tool.size();ii++)
	{
	  int capid = (*iho)[ii].capid();
	  if (subpedvalue_) fAdcSum+=(tool[ii]-calibrations.pedestal(capid));
	  if (!subpedvalue_) fAdcSum+=(tool[ii]-4.5);
	}
      ndigis++;
    }      
   
    if (menDigisHO) menDigisHO->Fill(ndigis);
    ndigis=0;
    edm::Handle<PCaloHitContainer> hcalHits ;
    iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
    
    const PCaloHitContainer * simhitResult = hcalHits.product () ;
    
    float fEnergySimHits = 0; 
    for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin () ;
	 simhits != simhitResult->end () ;
	 ++simhits)
      {    
	HcalDetId detId(simhits->id());
	//  3 == HO
	if (detId.subdet()== 3  ){  fEnergySimHits += simhits->energy(); }
      }
    
    if (meDigiSimhitHO) meDigiSimhitHO->Fill( fEnergySimHits, fAdcSum);
    if (meRatioDigiSimhitHO) meRatioDigiSimhitHO->Fill(fAdcSum/fEnergySimHits);
    if (meDigiSimhitHOprofile) meDigiSimhitHOprofile->Fill ( fEnergySimHits, fAdcSum);
 if (meSumDigisHO) meSumDigisHO->Fill(fAdcSum);
  }
    //  ******************** END of Hcal Outer *****************************
    
}


DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE (HcalDigiTester) ;
