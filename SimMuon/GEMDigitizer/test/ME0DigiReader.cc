/** \class ME0DigiReader
 *
 *  Dumps ME0 digis 
 *  
 *  \authors: Roumyana Hadjiiska
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include <map>
#include <set>

#include "DataFormats/Common/interface/DetSet.h"
#include <iostream>

#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TGraphErrors.h"

using namespace std;


class ME0DigiReader: public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:

  explicit ME0DigiReader(const edm::ParameterSet& pset);
  
  virtual ~ME0DigiReader(){}
  
  void beginJob();
  void analyze(const edm::Event &, const edm::EventSetup&); 
  void endJob();
  
private:

  edm::EDGetTokenT<edm::PSimHitContainer> simhitToken_;
  edm::EDGetTokenT<ME0DigiCollection> me0DigiToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > me0StripDigiSimLinkToken_;
  bool debug_; 

  TH1F *hProces; 
  TH2F *hNstripEtaParts;
  TH1F *hBx;
  TH2F *hRadiusEtaPartVsNdigi;
  TH2F *hRadiusEtaPartVsNdigiOvTrArea;
  TH1F *hRadiusEtaPart;
  TH1F *hdeltaXEntryPointVsCentreStrip;
  TH1F *hResidualsSimPhi;
  TH1F *hResidualsDigiPhi;
  TH1F *hResidualsSimVsDigiPhi;
  TGraphErrors *grRatePerRoll;

  int numbEvents;
  int ndigiEtaPart[8] = {0};
  double ndigiVsArea[8] = {0};
  double rollRadiusEtaPart[8] = {0};

};



ME0DigiReader::ME0DigiReader(const edm::ParameterSet& pset) :
  simhitToken_(consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("simhitToken"))),
  me0DigiToken_(consumes<ME0DigiCollection>(pset.getParameter<edm::InputTag>("me0DigiToken"))),
  me0StripDigiSimLinkToken_(consumes<edm::DetSetVector<StripDigiSimLink> >(pset.getParameter<edm::InputTag>("me0StripDigiSimLinkToken")))
  , debug_(pset.getParameter<bool>("debugFlag"))
{
  usesResource("TFileService");
  edm::Service < TFileService > fs; 

  hProces = fs->make < TH1F > ("hProces", "Process type for all the simHits", 20, 0, 20);
  hNstripEtaParts = fs->make <TH2F> ("NstripEtaParts", "Nstrips in all EtaPartitions",20, 0, 10, 770, 1, 770);
  hBx = fs->make <TH1F> ("hBx", "bx from digi - for all #eta partiotions", 9, -5.5, 3.5 );
  hRadiusEtaPartVsNdigi = fs->make <TH2F> ("hRadiusEtaPartVsNdigi", "Radius Eta Partition vs Ndigi", 2500, 0., 250., 200, 0., 20. ); 
  hRadiusEtaPartVsNdigiOvTrArea = fs->make <TH2F> ("hRadiusEtaPartVsNdigiOvTrArea", "Ndigi/TrArea vs Radius Eta Partition", 2500, 0., 250., 1000, 0., 0.1 );
  hRadiusEtaPart = fs->make <TH1F> ("hRadiusEtaPart", "Radius Eta Partition", 200, 0., 200. );
  hdeltaXEntryPointVsCentreStrip = fs->make <TH1F> ("deltaX", "delta X Residuals", 200, -10., 10. );
  hResidualsSimPhi= fs->make <TH1F> ("ResidualsSimPhi", "Global SimMuon Phi", 200, -10., 10. );
  hResidualsDigiPhi= fs->make <TH1F> ("ResidualsDigiPhi", "Global DigiMuon Phi", 200, -10., 10. );
  hResidualsSimVsDigiPhi= fs->make <TH1F> ("ResidualsSimVsDigiPhi", "Residuals (SimMuon-Digi) Phi", 50000, -0.5, 0.5 );

  grRatePerRoll = fs->make<TGraphErrors> (8);
  grRatePerRoll->SetName("grRatePerRoll");
  grRatePerRoll->SetTitle("ME0 Rate vs Roll Radius - BKG model");

  numbEvents = 0;
}

void ME0DigiReader::beginJob() {
}

void ME0DigiReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{

  edm::ESHandle<ME0Geometry> pDD;
  eventSetup.get<MuonGeometryRecord>().get( pDD );

  edm::Handle<edm::PSimHitContainer> simHits; 
  event.getByToken(simhitToken_, simHits);    

  edm::Handle<ME0DigiCollection> digis;
  event.getByToken(me0DigiToken_, digis);
   
  edm::Handle< edm::DetSetVector<StripDigiSimLink> > thelinkDigis;
  event.getByToken(me0StripDigiSimLinkToken_, thelinkDigis);

  ME0DigiCollection::DigiRangeIterator detUnitIt;

  int countRoll[8]={0};
  for (detUnitIt = digis->begin(); detUnitIt != digis->end(); ++detUnitIt)
  {
    const ME0DetId& id = (*detUnitIt).first;
    const ME0EtaPartition* roll = pDD->etaPartition(id);

      
    int ndigi = 0;
    double trArea(0.0);
    double trStripArea(0.0);
    Local3DPoint locMuonEntry(0., 0., 0.);
    GlobalPoint globMuonEntry(0., 0., 0.);
    double simMuPhi = -99.;
    double deltaPhi = -99.;
    Local3DPoint locDigi(0., 0., 0.);
    GlobalPoint pointDigiHit;

    const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));

    const float rollRadius = top_->radius();
    const float striplength(top_->stripLength());
    const int nstrips = roll->nstrips();
    trStripArea = (roll->pitch()) * striplength;
    trArea = trStripArea * nstrips;

    if(id.roll() == 1) { countRoll[0]++;}
    if(id.roll() == 2) { countRoll[1]++;}
    if(id.roll() == 3) { countRoll[2]++;}
    if(id.roll() == 4) { countRoll[3]++;}
    if(id.roll() == 5) { countRoll[4]++;}
    if(id.roll() == 6) { countRoll[5]++;}
    if(id.roll() == 7) { countRoll[6]++;}
    if(id.roll() == 8) { countRoll[7]++;}

    // Loop over the digis of this DetUnit
    const ME0DigiCollection::Range& range = (*detUnitIt).second;
    for (ME0DigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
    {
      if(id.roll() == 1) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[0]++;}
      if(id.roll() == 2) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[1]++;}
      if(id.roll() == 3) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[2]++;}
      if(id.roll() == 4) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[3]++;}
      if(id.roll() == 5) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[4]++;}
      if(id.roll() == 6) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[5]++;}
      if(id.roll() == 7) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[6]++;}
      if(id.roll() == 8) {hNstripEtaParts->Fill(id.roll(),digiIt->strip()); ndigiEtaPart[7]++;}

      //bx
      hBx->Fill(digiIt->bx()); 
        ndigi++;

      if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips() )
      {
        cout <<" XXXXXXXXXXXXX Problemt with "<<id<<"  a digi has strip# = "<<digiIt->strip()<<endl;
      }
 
      for(const auto& simHit: *simHits)
      {
	
	hProces->Fill(simHit.processType());
	
        ME0DetId me0Id(simHit.detUnitId());

        if (me0Id == id )//&& abs(simHit.particleType()) == 13)
        {

	  locMuonEntry = simHit.entryPoint();
	  globMuonEntry = roll->toGlobal(locMuonEntry);
	  simMuPhi = globMuonEntry.phi();
	  locDigi = roll->centreOfStrip(digiIt->strip());
	  pointDigiHit = roll->toGlobal(locDigi);
	  double digiPhi= pointDigiHit.phi();
	  deltaPhi = simMuPhi - digiPhi;
	  hdeltaXEntryPointVsCentreStrip->Fill(( simHit.entryPoint().x()-roll->centreOfStrip(digiIt->strip()).x() ));
	  hResidualsSimPhi->Fill(simMuPhi);
          hResidualsDigiPhi->Fill(digiPhi);
          hResidualsSimVsDigiPhi->Fill(deltaPhi);

        }

      }
    }// for digis in roll

    hRadiusEtaPartVsNdigi->Fill(rollRadius, ndigi);
    hRadiusEtaPartVsNdigiOvTrArea->Fill(rollRadius,ndigi/trArea);
    hRadiusEtaPart->Fill(rollRadius);

    if(id.roll() == 1) {ndigiVsArea[0] = ndigiVsArea[0] + ndigiEtaPart[0]*1./trArea; rollRadiusEtaPart[0] = rollRadius;}
    if(id.roll() == 2) {ndigiVsArea[1] = ndigiVsArea[1] + ndigiEtaPart[1]*1./trArea; rollRadiusEtaPart[1] = rollRadius;}
    if(id.roll() == 3) {ndigiVsArea[2] = ndigiVsArea[2] + ndigiEtaPart[2]*1./trArea; rollRadiusEtaPart[2] = rollRadius;}
    if(id.roll() == 4) {ndigiVsArea[3] = ndigiVsArea[3] + ndigiEtaPart[3]*1./trArea; rollRadiusEtaPart[3] = rollRadius;}
    if(id.roll() == 5) {ndigiVsArea[4] = ndigiVsArea[4] + ndigiEtaPart[4]*1./trArea; rollRadiusEtaPart[4] = rollRadius;}
    if(id.roll() == 6) {ndigiVsArea[5] = ndigiVsArea[5] + ndigiEtaPart[5]*1./trArea; rollRadiusEtaPart[5] = rollRadius;}
    if(id.roll() == 7) {ndigiVsArea[6] = ndigiVsArea[6] + ndigiEtaPart[6]*1./trArea; rollRadiusEtaPart[6] = rollRadius;}
    if(id.roll() == 8) {ndigiVsArea[7] = ndigiVsArea[7] + ndigiEtaPart[7]*1./trArea; rollRadiusEtaPart[7] = rollRadius;}

  }// for eta partitions (rolls)

  std::cout << "roll 1 numbers = " <<  countRoll[0] << "\tndigi = " << ndigiEtaPart[0] << std::endl;
  std::cout << "roll 2 numbers = " <<  countRoll[1] << "\tndigi = " << ndigiEtaPart[1] << std::endl;
  std::cout << "roll 3 numbers = " <<  countRoll[2] << "\tndigi = " << ndigiEtaPart[2] << std::endl;
  std::cout << "roll 4 numbers = " <<  countRoll[3] << "\tndigi = " << ndigiEtaPart[3] << std::endl;
  std::cout << "roll 5 numbers = " <<  countRoll[4] << "\tndigi = " << ndigiEtaPart[4] << std::endl;
  std::cout << "roll 6 numbers = " <<  countRoll[5] << "\tndigi = " << ndigiEtaPart[5] << std::endl;
  std::cout << "roll 7 numbers = " <<  countRoll[6] << "\tndigi = " << ndigiEtaPart[6] << std::endl;
  std::cout << "roll 8 numbers = " <<  countRoll[7] << "\tndigi = " << ndigiEtaPart[7]<< std::endl;

  numbEvents++;
}

void ME0DigiReader::endJob() {
  std::cout << "number of events = " << numbEvents << std::endl;
  std::cout << "--------------" << std::endl;
  //  hRadiusEtaPartVsNdigiOvTrArea->GetXProjections();

  std::vector<double> myRadii, myRates;

  std::cout << "ndigiVsArea1 = " << ndigiVsArea[0];
  ndigiVsArea[0] = ndigiVsArea[0]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[0] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[0]); myRates.push_back(ndigiVsArea[0]);

  std::cout << "ndigiVsArea2 = " << ndigiVsArea[1];
  ndigiVsArea[1] = ndigiVsArea[1]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[1] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[1]); myRates.push_back(ndigiVsArea[1]);

  std::cout << "ndigiVsArea3 = " << ndigiVsArea[2];
  ndigiVsArea[2] = ndigiVsArea[2]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[2] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[2]); myRates.push_back(ndigiVsArea[2]);

  std::cout << "ndigiVsArea4 = " << ndigiVsArea[3];
  ndigiVsArea[3] = ndigiVsArea[3]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[3] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[3]); myRates.push_back(ndigiVsArea[3]);

  std::cout << "ndigiVsArea5 = " << ndigiVsArea[4];
  ndigiVsArea[4] = ndigiVsArea[4]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[4] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[4]); myRates.push_back(ndigiVsArea[4]);

  std::cout << "ndigiVsArea6 = " << ndigiVsArea[5];
  ndigiVsArea[5] = ndigiVsArea[5]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[5] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[5]); myRates.push_back(ndigiVsArea[5]);

  std::cout << "ndigiVsArea7 = " << ndigiVsArea[6];
  ndigiVsArea[6] = ndigiVsArea[6]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[6] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[6]); myRates.push_back(ndigiVsArea[6]);

  std::cout << "ndigiVsArea8 = " << ndigiVsArea[7];
  ndigiVsArea[7] = ndigiVsArea[7]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea[7] << std::endl;

  myRadii.push_back(rollRadiusEtaPart[7]); myRates.push_back(ndigiVsArea[7]);

  std::cout << "rollRadius[cm]\tRate[Hz/cm2]" << std::endl;
  std::cout << rollRadiusEtaPart[0] << "\t" << ndigiVsArea[0] << std::endl;
  std::cout << rollRadiusEtaPart[1] << "\t" << ndigiVsArea[1] << std::endl;
  std::cout << rollRadiusEtaPart[2] << "\t" << ndigiVsArea[2] << std::endl;
  std::cout << rollRadiusEtaPart[3] << "\t" << ndigiVsArea[3] << std::endl;
  std::cout << rollRadiusEtaPart[4] << "\t" << ndigiVsArea[4] << std::endl;
  std::cout << rollRadiusEtaPart[5] << "\t" << ndigiVsArea[5] << std::endl;
  std::cout << rollRadiusEtaPart[6] << "\t" << ndigiVsArea[6] << std::endl;
  std::cout << rollRadiusEtaPart[7] << "\t" << ndigiVsArea[7] << std::endl;


  for (unsigned int i = 0; i < myRadii.size(); i++)
  {
    std::cout << "radius = " << myRadii[i] << "\tRate = " << myRates[i] << std::endl;
    grRatePerRoll->SetPoint(i, myRadii[i], myRates[i]);

  }

}

#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(ME0DigiReader);

