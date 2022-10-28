#include "SimG4Core/GFlash/TB/TreeMatrixCalib.h"

TreeMatrixCalib::TreeMatrixCalib(const char*) {
  edm::Service<TFileService> fs;
  myTree_ = fs->make<TTree>("T1", "my tree");

  // Amplitude / hodoscopes / tdc infos
  myTree_->Branch("run", &myRun, "run/I");
  myTree_->Branch("event", &myEvent, "event/I");
  myTree_->Branch("xtalSM", &myXtalSM, "xtalSM/I");
  myTree_->Branch("maxEneXtal", &myMaxEneXtal, "maxEneXtal/I");
  myTree_->Branch("nominalXtalSM", &myNominalXtalSM, "nominalXtalSM/I");
  myTree_->Branch("nextXtalSM", &myNextXtalSM, "nextXtalSM/I");
  myTree_->Branch("xtalEta", &myXtalEta, "xtalEta/I");
  myTree_->Branch("xtalPhi", &myXtalPhi, "xtalPhi/I");
  myTree_->Branch("tbMoving", &myTbMoving, "tbMoving/I");
  myTree_->Branch("hodoX", &myHodoX, "hodoX/D");
  myTree_->Branch("hodoY", &myHodoY, "hodoY/D");
  myTree_->Branch("caloX", &myCaloX, "caloX/D");
  myTree_->Branch("caloY", &myCaloY, "caloY/D");
  myTree_->Branch("hodoSlopeX", &myHodoSlopeX, "hodoSlopeX/D");
  myTree_->Branch("hodoSlopeY", &myHodoSlopeY, "hodoSlopeY/D");
  myTree_->Branch("hodoQualityX", &myHodoQualityX, "hodoQualityX/D");
  myTree_->Branch("hodoQualityY", &myHodoQualityY, "hodoQualityY/D");
  myTree_->Branch("tdcOffset", &myTdcOffset, "tdcOffset/D");
  myTree_->Branch("allMatrix", &myAllMatrix, "allMatrix/I");
  myTree_->Branch("amplit", &myAmplit, "amplit[49]/D");
  myTree_->Branch("crystal", &myCrystal, "crystal[49]/I");
}

void TreeMatrixCalib::store() { myTree_->Fill(); }

void TreeMatrixCalib::fillInfo(int run,
                               int eve,
                               int xnum,
                               int maxX,
                               int nomX,
                               int nextX,
                               int xeta,
                               int xphi,
                               int tbm,
                               double xx,
                               double yy,
                               double ecalx,
                               double ecaly,
                               double sx,
                               double sy,
                               double qx,
                               double qy,
                               double tdcoff,
                               int allm,
                               double amp[],
                               int cry[]) {
  myRun = run;
  myEvent = eve;
  myXtalSM = xnum;
  myMaxEneXtal = maxX;
  myNominalXtalSM = nomX;
  myNextXtalSM = nextX;
  myXtalEta = xeta;
  myXtalPhi = xphi;
  myTbMoving = tbm;
  myHodoX = xx;
  myHodoY = yy;
  myCaloX = ecalx;
  myCaloY = ecaly;
  myHodoSlopeX = sx;
  myHodoSlopeY = sy;
  myHodoQualityX = qx;
  myHodoQualityY = qy;
  myTdcOffset = tdcoff;
  myAllMatrix = allm;
  for (int ii = 0; ii < 49; ii++) {
    myAmplit[ii] = amp[ii];
    myCrystal[ii] = cry[ii];
  }
}
