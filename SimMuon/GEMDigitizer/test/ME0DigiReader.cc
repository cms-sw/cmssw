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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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

    if ( id.roll()>0 && id.roll()<9) countRoll[id.roll()-1]++;

    // Loop over the digis of this DetUnit
    const ME0DigiCollection::Range& range = (*detUnitIt).second;
    for (ME0DigiCollection::const_iterator digiIt = range.first; digiIt!=range.second; ++digiIt)
    {
      hNstripEtaParts->Fill(id.roll(),digiIt->strip());
      if ( id.roll()>0 && id.roll()<9) ndigiEtaPart[id.roll()-1]++;

      //bx
      hBx->Fill(digiIt->bx()); 
        ndigi++;

      if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips() )
      {
        LogDebug("ME0DigiReader") <<" XXXXXXXXXXXXX Problemt with "<<id<<"  a digi has strip# = "<<digiIt->strip()<<endl;
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

    if ( id.roll()>0 && id.roll()<9) {
      ndigiVsArea[id.roll()-1] = ndigiVsArea[id.roll()-1] + ndigiEtaPart[id.roll()-1]*1./trArea;
      rollRadiusEtaPart[id.roll()-1] = rollRadius;
    }
  }// for eta partitions (rolls)

  for (int i=0; i<=8; ++i){
    LogDebug("ME0DigiReader") << "roll "<<i+1<<" numbers = " <<  countRoll[i] << "\tndigi = " << ndigiEtaPart[i] << std::endl;
  }

  numbEvents++;
}

void ME0DigiReader::endJob() {
  LogDebug("ME0DigiReader") << "number of events = " << numbEvents << std::endl;
  LogDebug("ME0DigiReader") << "--------------" << std::endl;
  //  hRadiusEtaPartVsNdigiOvTrArea->GetXProjections();

  std::vector<double> myRadii, myRates;

  for (int i=0; i<=8; ++i){
    LogDebug("ME0DigiReader") << "ndigiVsArea"<<i+1<< " = " << ndigiVsArea[i];
    ndigiVsArea[0] = ndigiVsArea[i]/(numbEvents * 9 *25 * 1.0e-9 * 2. * 18.  * 6. * 1.5);
    LogDebug("ME0DigiReader") << "\tRate [Hz/cm2] = " << ndigiVsArea[i] << std::endl;
    myRadii.push_back(rollRadiusEtaPart[i]);
    myRates.push_back(ndigiVsArea[i]);
  }

  LogDebug("ME0DigiReader") << "rollRadius[cm]\tRate[Hz/cm2]" << std::endl;
  for (int i=0; i<=8; ++i){
    LogDebug("ME0DigiReader") << rollRadiusEtaPart[i] << "\t" << ndigiVsArea[i] << std::endl;
  }

  for (unsigned int i = 0; i < myRadii.size(); i++)
  {
    LogDebug("ME0DigiReader") << "radius = " << myRadii[i] << "\tRate = " << myRates[i] << std::endl;
    grRatePerRoll->SetPoint(i, myRadii[i], myRates[i]);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ME0DigiReader);

