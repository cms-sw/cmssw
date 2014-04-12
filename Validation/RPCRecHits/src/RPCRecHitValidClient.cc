#include "Validation/RPCRecHits/interface/RPCRecHitValidClient.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

typedef MonitorElement* MEP;

RPCRecHitValidClient::RPCRecHitValidClient(const edm::ParameterSet& pset)
{
  subDir_ = pset.getParameter<std::string>("subDir");
}

void RPCRecHitValidClient::endRun(const edm::Run& run, const edm::EventSetup& eventSetup)
{
  DQMStore* dbe = edm::Service<DQMStore>().operator->();
  if ( !dbe ) return;

  dbe->setCurrentFolder(subDir_);
  MEP me_rollEfficiencyBarrel_eff = dbe->book1D("RollEfficiencyBarrel_eff", "Roll efficiency in Barrel;Efficiency [%]", 50+2, -2, 100+2);
  MEP me_rollEfficiencyEndcap_eff = dbe->book1D("RollEfficiencyEndcap_eff", "Roll efficiency in Endcap;Efficiency [%]", 50+2, -2, 100+2);
  MEP me_rollEfficiencyStatCutOffBarrel_eff = dbe->book1D("RollEfficiencyCutOffBarrel_eff", "Roll efficiency in Barrel without low stat chamber;Efficiency [%]", 50+2, -2, 100+2);
  MEP me_rollEfficiencyStatCutOffEndcap_eff = dbe->book1D("RollEfficiencyCutOffEndcap_eff", "Roll efficiency in Endcap without low stat chamber;Efficiency [%]", 50+2, -2, 100+2);

  const double maxNoise = 1e-7;
  MEP me_rollNoiseBarrel_noise = dbe->book1D("RollNoiseBarrel_noise", "Roll noise in Barrel;Noise level [Event^{-1}cm^{-2}]", 25+2, -maxNoise/25, maxNoise+maxNoise/25);
  MEP me_rollNoiseEndcap_noise = dbe->book1D("RollNoiseEndcap_noise", "Roll noise in Endcap;Noise level [Event^{-1}cm^{-2}]", 25+2, -maxNoise/25, maxNoise+maxNoise/25);

  MEP me_matchOccupancyBarrel_detId = dbe->get(subDir_+"/Occupancy/MatchOccupancyBarrel_detId");
  MEP me_matchOccupancyEndcap_detId = dbe->get(subDir_+"/Occupancy/MatchOccupancyEndcap_detId");
  MEP me_refOccupancyBarrel_detId = dbe->get(subDir_+"/Occupancy/RefOccupancyBarrel_detId");
  MEP me_refOccupancyEndcap_detId = dbe->get(subDir_+"/Occupancy/RefOccupancyEndcap_detId");

  if ( me_matchOccupancyBarrel_detId and me_refOccupancyBarrel_detId )
  {
    TH1* h_matchOccupancyBarrel_detId = me_matchOccupancyBarrel_detId->getTH1();
    TH1* h_refOccupancyBarrel_detId = me_refOccupancyBarrel_detId->getTH1();

    for ( int bin = 1, nBin = h_matchOccupancyBarrel_detId->GetNbinsX(); bin <= nBin; ++bin )
    {
      const double nRec = h_matchOccupancyBarrel_detId->GetBinContent(bin);
      const double nRef = h_refOccupancyBarrel_detId->GetBinContent(bin);

      const double eff = nRef ? nRec/nRef*100 : -1;

      me_rollEfficiencyBarrel_eff->Fill(eff);
      if ( nRef >= 20 ) me_rollEfficiencyStatCutOffBarrel_eff->Fill(eff);
    }
  }

  if ( me_matchOccupancyEndcap_detId and me_refOccupancyEndcap_detId )
  {
    TH1* h_matchOccupancyEndcap_detId = me_matchOccupancyEndcap_detId->getTH1();
    TH1* h_refOccupancyEndcap_detId = me_refOccupancyEndcap_detId->getTH1();

    for ( int bin = 1, nBin = h_matchOccupancyEndcap_detId->GetNbinsX(); bin <= nBin; ++bin )
    {
      const double nRec = h_matchOccupancyEndcap_detId->GetBinContent(bin);
      const double nRef = h_refOccupancyEndcap_detId->GetBinContent(bin);

      const double eff = nRef ? nRec/nRef*100 : -1;

      me_rollEfficiencyEndcap_eff->Fill(eff);
      if ( nRef >= 20 ) me_rollEfficiencyStatCutOffEndcap_eff->Fill(eff);
    }
  }

  MEP me_eventCount = dbe->get(subDir_+"/Occupancy/EventCount");
  const double nEvent = me_eventCount ? me_eventCount->getTH1()->GetBinContent(1) : 1;
  MEP me_noiseOccupancyBarrel_detId = dbe->get(subDir_+"/Occupancy/NoiseOccupancyBarrel_detId");
  MEP me_rollAreaBarrel_detId = dbe->get(subDir_+"/Occupancy/RollAreaBarrel_detId");
  if ( me_noiseOccupancyBarrel_detId and me_rollAreaBarrel_detId )
  {
    TH1* h_noiseOccupancyBarrel_detId = me_noiseOccupancyBarrel_detId->getTH1();
    TH1* h_rollAreaBarrel_detId = me_rollAreaBarrel_detId->getTH1();

    for ( int bin = 1, nBin = h_noiseOccupancyBarrel_detId->GetNbinsX(); bin <= nBin; ++bin )
    {
      const double noiseCount = h_noiseOccupancyBarrel_detId->GetBinContent(bin);
      const double area = h_rollAreaBarrel_detId->GetBinContent(bin);
      const double noiseLevel = area > 0 ? noiseCount/area/nEvent : 0;
      if ( noiseLevel == 0. ) me_rollNoiseBarrel_noise->Fill(-maxNoise/50); // Fill underflow bin if noise is exactly zero
      else me_rollNoiseBarrel_noise->Fill(std::min(noiseLevel, maxNoise));
    }
  }

  MEP me_noiseOccupancyEndcap_detId = dbe->get(subDir_+"/Occupancy/NoiseOccupancyEndcap_detId");
  MEP me_rollAreaEndcap_detId = dbe->get(subDir_+"/Occupancy/RollAreaEndcap_detId");
  if ( me_noiseOccupancyEndcap_detId and me_rollAreaEndcap_detId )
  {
    TH1* h_noiseOccupancyEndcap_detId = me_noiseOccupancyEndcap_detId->getTH1();
    TH1* h_rollAreaEndcap_detId = me_rollAreaEndcap_detId->getTH1();

    for ( int bin = 1, nBin = h_noiseOccupancyEndcap_detId->GetNbinsX(); bin <= nBin; ++bin )
    {
      const double noiseCount = h_noiseOccupancyEndcap_detId->GetBinContent(bin);
      const double area = h_rollAreaEndcap_detId->GetBinContent(bin);
      const double noiseLevel = area > 0 ? noiseCount/area/nEvent : 0;
      if ( noiseLevel == 0 ) me_rollNoiseEndcap_noise->Fill(-maxNoise/50); // Fill underflow bin if noise if exactly zero
      else me_rollNoiseEndcap_noise->Fill(std::min(noiseLevel, maxNoise));
    }
  }

}

DEFINE_FWK_MODULE(RPCRecHitValidClient);

