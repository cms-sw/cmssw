#include "Validation/RPCRecHits/interface/RPCValidHistograms.h"

#include "TAxis.h"

void RPCValidHistograms::bookHistograms(DQMStore* dbe, const std::string subDir)
{
  if ( !dbe ) 
  {
    edm::LogError("RPCValidHistograms") << "DBE not initialized\n";
    return;
  }
  if ( booked_ )
  {
    edm::LogError("RPCValidHistograms") << "Histogram is already booked\n";
    return;
  }

  const std::string pwd = dbe->pwd();
  dbe->setCurrentFolder(subDir);

  // Book histograms
  clusterSize = dbe->book1D("ClusterSize", "Cluster size;Cluster size", 11, -0.5, 10.5);

  // Number of hits
  dbe->setCurrentFolder(subDir+"/Occupancy");
  nRefHit_W = dbe->book1D("NRefHit_Wheel", "Number of reference Hits;Wheel", 5, -2.5, 2.5);
  nRefHit_D = dbe->book1D("NRefHit_Disk", "Number of reference Hits;Disk", 7, -3.5, 3.5);
  nRecHit_W = dbe->book1D("NRecHit_Wheel", "Number of recHits;Wheel", 5, -2.5, 2.5);
  nRecHit_D = dbe->book1D("NRecHit_Disk", "Number of recHits;Disk", 7, -3.5, 3.5);

  nRefHit_WvsR = dbe->book2D("NRefHit_WvsR", "Number of reference Hits;Wheel;Station", 5, -2.5, 2.5, 4, 1, 5);
  nRecHit_WvsR = dbe->book2D("NRecHit_WvsR", "Number of recHits;Wheel;Station", 5, -2.5, 2.5, 4, 1, 5);
  nRefHit_DvsR = dbe->book2D("NRefHit_DvsR", "Number of reference Hits;Disk;Ring", 7, -3.5, 3.5, 4, 1, 5);
  nRecHit_DvsR = dbe->book2D("NRecHit_DvsR", "Number of recHits;Disk;Ring", 7, -3.5, 3.5, 4, 1, 5);

  nMatchedRefHit_W = dbe->book1D("NMatchedRefHit_Wheel", "Number of matched reference Hits;Wheel", 5, -2.5, 2.5);
  nMatchedRefHit_D = dbe->book1D("NMatchedRefHit_Disk", "Number of matched reference Hits;Disk", 7, -3.5, 3.5);

  nMatchedRefHit_WvsR = dbe->book2D("NMatchedRefHit_WvsR", "Number of matched reference Hits;Wheel;Station", 5, -2.5, 2.5, 4, 1, 5);
  nMatchedRefHit_DvsR = dbe->book2D("NMatchedRefHit_DvsR", "Number of matched reference Hits;Disk;Ring", 7, -3.5, 3.5, 4, 1, 5);

  nUnMatchedRefHit_W = dbe->book1D("NUnMatchedRefHit_Wheel", "Number of un-matched reference Hits;Wheel", 5, -2.5, 2.5);
  nUnMatchedRefHit_D = dbe->book1D("NUnMatchedRefHit_Disk", "Number of un-matched reference Hits;Disk", 7, -3.5, 3.5);
  nUnMatchedRecHit_W = dbe->book1D("NUnMatchedRecHit_Wheel", "Number of un-matched recHits;Wheel", 5, -2.5, 2.5);
  nUnMatchedRecHit_D = dbe->book1D("NUnMatchedRecHit_Disk", "Number of un-matched recHits;Disk", 7, -3.5, 3.5);

  nUnMatchedRefHit_WvsR = dbe->book2D("NUnMatchedRefHit_WvsR", "Number of un-matched reference Hits;Wheel;Station", 5, -2.5, 2.5, 4, 1, 5);
  nUnMatchedRecHit_WvsR = dbe->book2D("NUnMatchedRecHit_WvsR", "Number of un-matched recHits;Wheel;Station", 5, -2.5, 2.5, 4, 1, 5);
  nUnMatchedRefHit_DvsR = dbe->book2D("NUnMatchedRefHit_DvsR", "Number of un-matched reference Hits;Disk;Ring", 7, -3.5, 3.5, 4, 1, 5);
  nUnMatchedRecHit_DvsR = dbe->book2D("NUnMatchedRecHit_DvsR", "Number of un-matched recHits;Disk;Ring", 7, -3.5, 3.5, 4, 1, 5);

  // Residuals
  dbe->setCurrentFolder(subDir+"/Residual");
  res_W = dbe->book1D("Res_W", "Global Residuals for Wheel;Residual [cm]", 100, -8, 8);
  res_D = dbe->book1D("Res_D", "Global Residuals for Disk;Residual [cm]", 100, -8, 8);

  res2_W = dbe->book2D("Res2_W", "Residuals for Wheel;Wheel;Residual [cm]", 5, -2.5, 2.5, 50, -8, 8);
  res2_D = dbe->book2D("Res2_D", "Residuals for Disk;Disk;Residual [cm]", 7, -3.5, 3.5, 50, -8, 8);

  res2_WR = dbe->book2D("Res2_WR", "Redisuals for Wheel;Station;Residual [cm]", 4, 0, 4, 50, -8, 8);
  res2_DR = dbe->book2D("Res2_DR", "Redisuals for Disk;Ring;Residual [cm]", 4, 1, 5, 50, -8, 8);

  // Pulls
  pull_W = dbe->book1D("Pull_W", "Pull for Wheel;Pull", 100, -3, 3);
  pull_D = dbe->book1D("Pull_D", "Pull for Disk;Pull", 100, -3, 3);

  pull2_W = dbe->book2D("Pull2_W", "Pull for Wheel;Wheel;Pull", 5, -2.5, 2.5, 50, -3, 3);
  pull2_D = dbe->book2D("Pull2_D", "Pull for Disk;Disk;Pull", 7, -3.5, 3.5, 50, -3, 3);

  pull2_WR = dbe->book2D("Pull2_WR", "Pull for Wheel;Station;Pull", 4, 0, 4, 50, -3, 3);
  pull2_DR = dbe->book2D("Pull2_DR", "Pull for Disk;Ring;Pull", 4, 1, 5, 50, -3, 3);

  // Set bin labels
  for ( int i=1; i<=5; ++i )
  {
    TString binLabel = Form("Wheel %d", i-3);

    nRefHit_W->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nRecHit_W->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nRefHit_WvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nRecHit_WvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nMatchedRefHit_W->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nMatchedRefHit_WvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRefHit_W->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRecHit_W->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRefHit_WvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRecHit_WvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);

    res2_W->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    pull2_W->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=7; ++i )
  {
    TString binLabel = Form("Disk %d", i-4);

    nRefHit_D->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nRecHit_D->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nRefHit_DvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nRecHit_DvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nMatchedRefHit_D->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nMatchedRefHit_DvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRefHit_D->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRecHit_D->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRefHit_DvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRecHit_DvsR->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);

    res2_D->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    pull2_D->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Station %d", i);

    nRefHit_WvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nRecHit_WvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nMatchedRefHit_WvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRefHit_WvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRecHit_WvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Ring %d", i);

    nRefHit_DvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nRecHit_DvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nMatchedRefHit_DvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRefHit_DvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    nUnMatchedRecHit_DvsR->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }

  dbe->setCurrentFolder(pwd);
  booked_ = true;
}

