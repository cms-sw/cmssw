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

//  TH2F *hNstripEtaParts;
  TH1F *hNstripEtaPart1;
  TH1F *hNstripEtaPart2;
  TH1F *hNstripEtaPart3;
  TH1F *hNstripEtaPart4;
  TH1F *hNstripEtaPart5;
  TH1F *hNstripEtaPart6;
  TH1F *hNstripEtaPart7;
  TH1F *hNstripEtaPart8;
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
  int ndigi1, ndigi2, ndigi3, ndigi4, ndigi5, ndigi6, ndigi7, ndigi8;
  double ndigiVsArea1, ndigiVsArea2, ndigiVsArea3, ndigiVsArea4, ndigiVsArea5, ndigiVsArea6, ndigiVsArea7, ndigiVsArea8;
  double rollRadius1, rollRadius2, rollRadius3, rollRadius4, rollRadius5, rollRadius6, rollRadius7, rollRadius8;

};



ME0DigiReader::ME0DigiReader(const edm::ParameterSet& pset) :
  simhitToken_(consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("simhitToken"))),
  me0DigiToken_(consumes<ME0DigiCollection>(pset.getParameter<edm::InputTag>("me0DigiToken"))),
  me0StripDigiSimLinkToken_(consumes<edm::DetSetVector<StripDigiSimLink> >(pset.getParameter<edm::InputTag>("me0StripDigiSimLinkToken")))
  , debug_(pset.getParameter<bool>("debugFlag"))
{
  edm::Service < TFileService > fs; 

  hProces = fs->make < TH1F > ("hProces", "Process type for all the simHits", 20, 0, 20);
//  hNstripEtaParts = fs->make <TH2F> ("NstripEtaParts", "Nstrips in each EtaPartition ", 40, 0.5, 10.5, 770, 1, 770);
  hNstripEtaPart1 = fs->make <TH1F> ("NstripEtaPart1", "Nstrips in EtaPartition 1", 770, 1, 770);
  hNstripEtaPart2 = fs->make <TH1F> ("NstripEtaPart2", "Nstrips in EtaPartition 2", 770, 1, 770);
  hNstripEtaPart3 = fs->make <TH1F> ("NstripEtaPart3", "Nstrips in EtaPartition 3", 770, 1, 770);
  hNstripEtaPart4 = fs->make <TH1F> ("NstripEtaPart4", "Nstrips in EtaPartition 4", 770, 1, 770);
  hNstripEtaPart5 = fs->make <TH1F> ("NstripEtaPart5", "Nstrips in EtaPartition 5", 770, 1, 770);
  hNstripEtaPart6 = fs->make <TH1F> ("NstripEtaPart6", "Nstrips in EtaPartition 6", 770, 1, 770);
  hNstripEtaPart7 = fs->make <TH1F> ("NstripEtaPart7", "Nstrips in EtaPartition 7", 770, 1, 770);
  hNstripEtaPart8 = fs->make <TH1F> ("NstripEtaPart8", "Nstrips in EtaPartition 8", 770, 1, 770);
  hBx = fs->make <TH1F> ("hBx", "bx from digi - for all #eta partiotions", 9, -5.5, 3.5 );
  hRadiusEtaPartVsNdigi = fs->make <TH2F> ("hRadiusEtaPartVsNdigi", "Radius Eta Partition vs Ndigi", 2500, 0., 250., 200, 0., 20. );//MM 
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

  ndigi1 = 0; ndigi2 = 0; ndigi3 = 0; ndigi4 = 0; ndigi5 = 0; ndigi6 = 0; ndigi7 = 0; ndigi8 = 0;
  ndigiVsArea1 = 0.; ndigiVsArea2 = 0.; ndigiVsArea3 = 0.; ndigiVsArea4 = 0.; ndigiVsArea5 = 0.; ndigiVsArea6 = 0.; ndigiVsArea7 = 0.; ndigiVsArea8 = 0.;
  rollRadius1 = 0.; rollRadius2 = 0.; rollRadius3 = 0.; rollRadius4 = 0.; rollRadius5 = 0.; rollRadius6 = 0.; rollRadius7 = 0.; rollRadius8 = 0.; 

}

void ME0DigiReader::beginJob() {
}

void ME0DigiReader::analyze(const edm::Event & event, const edm::EventSetup& eventSetup)
{
//  cout << "--- Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  edm::ESHandle<ME0Geometry> pDD;
  eventSetup.get<MuonGeometryRecord>().get( pDD );

  edm::Handle<edm::PSimHitContainer> simHits; 
  event.getByToken(simhitToken_, simHits);    

  edm::Handle<ME0DigiCollection> digis;
  event.getByToken(me0DigiToken_, digis);
   
  edm::Handle< edm::DetSetVector<StripDigiSimLink> > thelinkDigis;
  event.getByToken(me0StripDigiSimLinkToken_, thelinkDigis);

  ME0DigiCollection::DigiRangeIterator detUnitIt;

  int countRoll1 = 0;
  int countRoll2 = 0;
  int countRoll3 = 0;
  int countRoll4 = 0;
  int countRoll5 = 0;
  int countRoll6 = 0;
  int countRoll7 = 0;
  int countRoll8 = 0;

  for (detUnitIt = digis->begin(); detUnitIt != digis->end(); ++detUnitIt)
  {
    const ME0DetId& id = (*detUnitIt).first;
    const ME0EtaPartition* roll = pDD->etaPartition(id);

    // ME0DetId print-out
//    cout<<"--------------"<<endl;
//      cout<<"id: "<<id.rawId()<<" etaPartition id.roll() = "<<id.roll()<<endl;
      
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

if(id.roll() == 1) { countRoll1++;}
if(id.roll() == 2) { countRoll2++;}
if(id.roll() == 3) { countRoll3++;}
if(id.roll() == 4) { countRoll4++;}
if(id.roll() == 5) { countRoll5++;}
if(id.roll() == 6) { countRoll6++;}
if(id.roll() == 7) { countRoll7++;}
if(id.roll() == 8) { countRoll8++;}

    // Loop over the digis of this DetUnit
    const ME0DigiCollection::Range& range = (*detUnitIt).second;
    for (ME0DigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
    {
/*
      for(int i = 0; i < id.roll(); i++){
	hNstripEtaParts->Fill(id.roll(), digiIt->strip());
      }
*/
      if(id.roll() == 1) {hNstripEtaPart1->Fill(digiIt->strip()); ndigi1++;}
      if(id.roll() == 2) {hNstripEtaPart2->Fill(digiIt->strip()); ndigi2++;}
      if(id.roll() == 3) {hNstripEtaPart3->Fill(digiIt->strip()); ndigi3++;}
      if(id.roll() == 4) {hNstripEtaPart4->Fill(digiIt->strip()); ndigi4++;}
      if(id.roll() == 5) {hNstripEtaPart5->Fill(digiIt->strip()); ndigi5++;}
      if(id.roll() == 6) {hNstripEtaPart6->Fill(digiIt->strip()); ndigi6++;}
      if(id.roll() == 7) {hNstripEtaPart7->Fill(digiIt->strip()); ndigi7++;}
      if(id.roll() == 8) {hNstripEtaPart8->Fill(digiIt->strip()); ndigi8++;}

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

    hRadiusEtaPartVsNdigi->Fill(rollRadius, ndigi);//MM
    hRadiusEtaPartVsNdigiOvTrArea->Fill(rollRadius,ndigi/trArea);//MM
    hRadiusEtaPart->Fill(rollRadius);

    if(id.roll() == 1) {ndigiVsArea1 = ndigiVsArea1 + ndigi1*1./trArea; rollRadius1 = rollRadius;}
    if(id.roll() == 2) {ndigiVsArea2 = ndigiVsArea2 + ndigi2*1./trArea; rollRadius2 = rollRadius;}
    if(id.roll() == 3) {ndigiVsArea3 = ndigiVsArea3 + ndigi3*1./trArea; rollRadius3 = rollRadius;}
    if(id.roll() == 4) {ndigiVsArea4 = ndigiVsArea4 + ndigi4*1./trArea; rollRadius4 = rollRadius;}
    if(id.roll() == 5) {ndigiVsArea5 = ndigiVsArea5 + ndigi5*1./trArea; rollRadius5 = rollRadius;}
    if(id.roll() == 6) {ndigiVsArea6 = ndigiVsArea6 + ndigi6*1./trArea; rollRadius6 = rollRadius;}
    if(id.roll() == 7) {ndigiVsArea7 = ndigiVsArea7 + ndigi7*1./trArea; rollRadius7 = rollRadius;}
    if(id.roll() == 8) {ndigiVsArea8 = ndigiVsArea8 + ndigi8*1./trArea; rollRadius8 = rollRadius;}

  }// for eta partitions (rolls)

std::cout << "roll 1 numbers = " <<  countRoll1 << "\tndigi = " << ndigi1 << std::endl;
std::cout << "roll 2 numbers = " <<  countRoll2 << "\tndigi = " << ndigi2 << std::endl;
std::cout << "roll 3 numbers = " <<  countRoll3 << "\tndigi = " << ndigi3 << std::endl;
std::cout << "roll 4 numbers = " <<  countRoll4 << "\tndigi = " << ndigi4 << std::endl;
std::cout << "roll 5 numbers = " <<  countRoll5 << "\tndigi = " << ndigi5 << std::endl;
std::cout << "roll 6 numbers = " <<  countRoll6 << "\tndigi = " << ndigi6 << std::endl;
std::cout << "roll 7 numbers = " <<  countRoll7 << "\tndigi = " << ndigi7 << std::endl;
std::cout << "roll 8 numbers = " <<  countRoll8 << "\tndigi = " << ndigi8 << std::endl;

/*
  for (edm::DetSetVector<StripDigiSimLink>::const_iterator itlink = thelinkDigis->begin(); itlink != thelinkDigis->end(); itlink++)
  {
    for(edm::DetSet<StripDigiSimLink>::const_iterator link_iter=itlink->data.begin();link_iter != itlink->data.end();++link_iter)
    {
      int detid = itlink->detId();
      int ev = link_iter->eventId().event();
      float frac =  link_iter->fraction();
      int strip = link_iter->channel();
      int trkid = link_iter->SimTrackId();
      int bx = link_iter->eventId().bunchCrossing();
      cout<<"DetUnit: "<<ME0DetId(detid)<<"  Event ID: "<<ev<<"  trkId: "<<trkid<<"  Strip: "<<strip<<"  Bx: "<<bx<<"  frac: "<<frac<<endl;
    }
  }
*/
  numbEvents++;
}

void ME0DigiReader::endJob() {
  std::cout << "number of events = " << numbEvents << std::endl;
  std::cout << "--------------" << std::endl;
//  hRadiusEtaPartVsNdigiOvTrArea->GetXProjections();

  std::vector<double> myRadii, myRates;

  std::cout << "ndigiVsArea1 = " << ndigiVsArea1;
  ndigiVsArea1 = ndigiVsArea1/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea1 << std::endl;

  myRadii.push_back(rollRadius1); myRates.push_back(ndigiVsArea1);

  std::cout << "ndigiVsArea2 = " << ndigiVsArea2;
  ndigiVsArea2 = ndigiVsArea2/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea2 << std::endl;

  myRadii.push_back(rollRadius2); myRates.push_back(ndigiVsArea2);

  std::cout << "ndigiVsArea3 = " << ndigiVsArea3;
  ndigiVsArea3 = ndigiVsArea3/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea3 << std::endl;

  myRadii.push_back(rollRadius3); myRates.push_back(ndigiVsArea3);

  std::cout << "ndigiVsArea4 = " << ndigiVsArea4;
  ndigiVsArea4 = ndigiVsArea4/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea4 << std::endl;

  myRadii.push_back(rollRadius4); myRates.push_back(ndigiVsArea4);

  std::cout << "ndigiVsArea5 = " << ndigiVsArea5;
  ndigiVsArea5 = ndigiVsArea5/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea5 << std::endl;

  myRadii.push_back(rollRadius5); myRates.push_back(ndigiVsArea5);

  std::cout << "ndigiVsArea6 = " << ndigiVsArea6;
  ndigiVsArea6 = ndigiVsArea6/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea6 << std::endl;

  myRadii.push_back(rollRadius6); myRates.push_back(ndigiVsArea6);

  std::cout << "ndigiVsArea7 = " << ndigiVsArea7;
  ndigiVsArea7 = ndigiVsArea7/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea7 << std::endl;

  myRadii.push_back(rollRadius7); myRates.push_back(ndigiVsArea7);

  std::cout << "ndigiVsArea8 = " << ndigiVsArea8;
  ndigiVsArea8 = ndigiVsArea8/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18. * 1.5);
  std::cout << "\tRate [Hz/cm2] = " << ndigiVsArea8 << std::endl;

  myRadii.push_back(rollRadius8); myRates.push_back(ndigiVsArea8);

  std::cout << "rollRadius[cm]\tRate[Hz/cm2]" << std::endl;
  std::cout << rollRadius1 << "\t" << ndigiVsArea1 << std::endl;
  std::cout << rollRadius2 << "\t" << ndigiVsArea2 << std::endl;
  std::cout << rollRadius3 << "\t" << ndigiVsArea3 << std::endl;
  std::cout << rollRadius4 << "\t" << ndigiVsArea4 << std::endl;
  std::cout << rollRadius5 << "\t" << ndigiVsArea5 << std::endl;
  std::cout << rollRadius6 << "\t" << ndigiVsArea6 << std::endl;
  std::cout << rollRadius7 << "\t" << ndigiVsArea7 << std::endl;
  std::cout << rollRadius8 << "\t" << ndigiVsArea8 << std::endl;


  for (unsigned int i = 0; i < myRadii.size(); i++)
  {
    std::cout << "radius = " << myRadii[i] << "\tRate = " << myRates[i] << std::endl;
    grRatePerRoll->SetPoint(i, myRadii[i], myRates[i]);

  }

}

#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(ME0DigiReader);

