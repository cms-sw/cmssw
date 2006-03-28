#include "/localscratch/dkonst/WORK/test/CMSSW_0_6_0_pre1/src/Validation/HcalDigis/interface/HcalDigiTester.h"


HcalDigiTester::HcalDigiTester(const edm::ParameterSet& iConfig)
{
}

HcalDigiTester::~HcalDigiTester()
{}

void HcalDigiTester::endJob() {

  myFile->cd();

  // Writing histograms
  //  h->Write();
  hEtaHB->Write();
  hPhiHB->Write();
  hDigiVsSim->Write();
  myFile->Close();

}
void HcalDigiTester::beginJob(const edm::EventSetup& c){

 myFile = new TFile ("hcalDigiTest.root","RECREATE");
 // Histograms
 hEtaHB =  new TH1F ("eta of digis(HB)","eta", 100, -5., 5.);
 hPhiHB =  new TH1F ("phi of digis(HB)","phi", 100, -3.14 , 3.14);
 hDigiVsSim = new TH2F("digis vs simhits(HB) ","digisvsihhits", 100, 0., 150., 100, 0.,0.2);
}

void
HcalDigiTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  // Get the digis
  // -------------
  edm::Handle<HBHEDigiCollection> hbhedigi ;
  iEvent.getByType (hbhedigi) ;

  edm::ESHandle<CaloGeometry> geometry ;
  iSetup.get<IdealGeometryRecord> ().get (geometry) ;
  
  // loop over the digis
  int ndigis=0;
  
  HBHEDigiCollection::const_iterator i;

  // ADC2fC
  edm::ESHandle<HcalDbService> conditions;
  iSetup.get<HcalDbRecord>().get(conditions);
  const HcalQIEShape* shape = conditions->getHcalShape();
  HcalCalibrations calibrations;
  

  CaloSamples tool;
  float fAdcSum=0; // sum of all ADC counts in terms of fC

  for (i=hbhedigi->begin();i!=hbhedigi->end();i++){
    HcalDetId cell(i->id());
    if (cell.subdet()==1  ) {
      const CaloCellGeometry* cellGeometry =
	geometry->getSubdetectorGeometry (cell)->getGeometry (cell) ;
      double fEta = cellGeometry->getPosition ().eta () ;
      double fPhi = cellGeometry->getPosition ().phi () ;
      hEtaHB->Fill(fEta);
      hPhiHB->Fill(fPhi);
      
      conditions->makeHcalCalibration(cell, &calibrations);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*i,tool);
      
      
      for  (int ii=0;ii<tool.size();ii++)
	{
	  int capid = (*i)[ii].capid();
	  fAdcSum=fAdcSum+(tool[ii]-calibrations.pedestal(capid));	 
	  //	std::cout<<" calibrations.pedestal(capid = "<<capid<< ") = "  <<  calibrations.pedestal(capid) <<std::endl;
	}
      ndigis++;
    }
  }
     
 
 edm::Handle<PCaloHitContainer> hcalHits ;
 iEvent.getByLabel("SimG4Object","HcalHits",hcalHits);
 

 const PCaloHitContainer * simhitResult = hcalHits.product () ;
 
 float fEnergySimHitsHB = 0; 
 for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin () ;
      simhits != simhitResult->end () ;
      ++simhits)
   {    
     HcalDetId detId(simhits->id());
     //  1 == HB
     if (detId.subdet()==1  ){  fEnergySimHitsHB = fEnergySimHitsHB + simhits->energy(); }
   }


 hDigiVsSim->Fill(fAdcSum , fEnergySimHitsHB);
}

DEFINE_SEAL_MODULE ();
DEFINE_ANOTHER_FWK_MODULE (HcalDigiTester) ;
