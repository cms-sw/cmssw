#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
  
class HexGeometry {

public :
  HexGeometry(bool fine);
  virtual ~HexGeometry() {}

  void initCellGeom(bool fine);
  void initWaferGeom();
  
  std::pair<double,double> position_cell(const int cell);
  std::pair<double,double> position_wafer(const int wafer);
  std::pair<double,double> cellGC(const int cell, const int wafer);


private :
  std::vector<std::pair<double,double> > xypos_cell;
  std::vector<std::pair<double,double> > xypos_wafer;

};

HexGeometry::HexGeometry(bool fine) {

  initCellGeom(fine);
  initWaferGeom();

}


/////cell geometry
void HexGeometry::initCellGeom(bool fine){
  const int nC(15), nF(20);
  //int nCoarse(11), nyCoarse(-42), nFine(15), nyFine(-56);
  int nCoarse(11), nyCoarse(21), nFine(15), nyFine(28);
  int cellCoarse[nC] = {2,5,8,11,12,11,12,11,12,11,12,11,8,5,2};
  int cellFine[nF] = {3,6,9,12,15,16,15,16,15,16,15,16,15,16,15,14,11,8,5,2};
  double wafer(123.7);

  int    rows = (fine) ? nF : nC;
  double cell = (fine) ? wafer/nFine : wafer/nCoarse;
  double dx   = 0.5*cell;
  //double dy   = 0.5*dx*tan(30.0*M_PI/180.0);
  double dy   = 0.5*cell*tan(30.0*M_PI/180.0);
  int    ny   = (fine) ? nyFine : nyCoarse;

  
  for (int ir = 0; ir < rows; ++ir) {
    int    column = (fine) ? cellFine[ir] : cellCoarse[ir];
    int    nx     = 1 - column;
    double ypos   = dy*ny;
    for (int ic = 0; ic<column; ++ic) {
      double xpos = dx*nx;
      nx += 2;
      //xypos.push_back(std::pair<double,double>(xpos,ypos));
      xypos_cell.push_back(std::pair<double,double>(ypos,xpos));  ///currently we have rotated the geometry by 90 degrees so x becomes y and y becomes x.
    }
    //ny += 6;
    ny -= 3;
  }//for (int ir = 0; ir < rows; ++ir)

  std::cout << "Initialize HexGeometry for " << xypos_cell.size() << " cells"
	    << std::endl;
  /*
  for (int ir = 0; ir < 133; ++ir) {
    std::cout << "ir : X : Y : " << ir << " " << xypos_cell[ir].first << " " << xypos_cell[ir].second << std::endl;
  }
  */
}


/////wafer geometry
void HexGeometry::initWaferGeom(){
  const int nwafer = 7;
  const int wafer[nwafer] = {0, 110101, 100101, 10002, 2, 10101, 101};
  const double wafersize = 124.7; // 123.7 mm + 1 mm gap
//const int wafer[nwafer] = {110101, 100101, 10002, 0, 2, 10101, 101};
//const double wafersize = 123.7; 

  double dy   = 0.5*wafersize;
  double dx   = 1.5*wafersize*tan(30.0*M_PI/180.0);
  for (int i = 0; i < nwafer; ++i) {
    int ix = (wafer[i]/100) % 100;
    if (((wafer[i]/100000)%10) == 0) ix = -ix;
    int iy = wafer[i] % 100;
    if (((wafer[i]/10000)%10) == 1)  iy = -iy;
    double xx = ix * dx;
    double yy = iy * dy;
//  std::cout << "wafer[" << i << "] " << wafer[i] << ":" << ix << ":" << iy
//	      << " (" << xx << ", " << yy << ")\n";
    xypos_wafer.push_back(std::pair<double,double>(xx,yy));
  }
  std::cout << "Initialize HexGeometry for " << xypos_wafer.size() << " wafers"
	    << std::endl;
  /*
  for (unsigned int ir = 0; ir < xypos_wafer.size(); ++ir) {
    std::cout << "ir : X : Y : " << ir << " " << xypos_wafer[ir].first << " " << xypos_wafer[ir].second << std::endl;
  }
  */


}


std::pair<double,double> HexGeometry::position_cell(const int cell) {
  std::pair<double,double> xy;
  if (cell >= 0 && cell < (int)(xypos_cell.size())) {
    xy = xypos_cell[cell];
  } else {
    xy = std::pair<double,double>(0,0);
  }
  return xy;
}


std::pair<double,double> HexGeometry::position_wafer(const int wafer) {
  std::pair<double,double> xy;
  if (wafer >= 0 && wafer < (int)(xypos_wafer.size())) {
    xy = xypos_wafer[wafer];
  } else {
    xy = std::pair<double,double>(0,0);
  }

//std::cout << "Wafer position for wafer # " << wafer << " are x and y " << xy.first << " " << xy.second << std::endl;
  return xy;
}

///returns the cell global coordinates if wafer no. and cell no. is known
std::pair<double,double> HexGeometry::cellGC(const int cell, const int wafer) {
  std::pair<double,double> xy;
  if (wafer >= 0 && wafer < (int)(xypos_wafer.size())) {
    
    double x = position_cell(cell).first + position_wafer(wafer).first;
    double y = position_cell(cell).second + position_wafer(wafer).second;
    
    xy = std::pair<double,double>(x,y);
  } else {
    xy = std::pair<double,double>(0,0);
  }
  std::cout << "X : Y : " << xy.first << " " << xy.second << std::endl;
  return xy;
}





void testGeometry() {

  HexGeometry geomc(false);
  for (int k = 0; k < 133; ++k) {
    std::pair<double,double> xy = geomc.position_cell(k);
    std::cout << "Coarse Cell[" << k << "] " << xy.first << ":" << xy.second
	      << std::endl;
  }


  for (int k = 0; k < 7; ++k) {
    std::pair<double,double> xy_w = geomc.position_wafer(k);
    std::cout << "Wafer[" << k << "] " << xy_w.first << ":" << xy_w.second
	      << std::endl;
  }

  /*HexGeometry geomf(true);
  for (int k = 0; k < 240; ++k) {
    std::pair<double,double> xy = geomf.position(k);
    std::cout << "Fine Cell[" << k << "] " << xy.first << ":" << xy.second
	      << std::endl;
  }
  */


}
