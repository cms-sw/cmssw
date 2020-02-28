#ifndef HCALTBEVENTPOSITION_H
#define HCALTBEVENTPOSITION_H 1

#include <string>
#include <iostream>
#include <vector>

/** \class HcalTBEventPosition


This class contains event position information, including the table
position as well as hits from the delay wire chambers.
      
$Date: 2006/04/04 15:00:27 $
$Revision: 1.3 $
\author P. Dudero - Minnesota
*/
class HcalTBEventPosition {
public:
  /// Null constructor
  HcalTBEventPosition();

  /// Get the X position (mm) of the HF table (if present in this run)
  double hfTableX() const { return hfTableX_; }
  /// Get the Y position (mm) of the HF table (if present in this run)
  double hfTableY() const { return hfTableY_; }
  /// Get the V position of the HF table (if present in this run)
  double hfTableV() const { return hfTableV_; }
  /// Get the eta (not ieta) position of the HB/HE/HO table (if present in this run)
  double hbheTableEta() const { return hbheTableEta_; }
  /// Get the phi (not iphi) position of the HB/HE/HO table (if present in this run)
  double hbheTablePhi() const { return hbheTablePhi_; }

  /** \brief Get the wire chamber hits for the specified chamber
      For HB/HE/HO running, chambers A, B, and C were active while
      all five (A, B, C, D, and E) were active for HF running.
  */
  void getChamberHits(char chamberch,  // 'A','B','C','D', or 'E'
                      std::vector<double>& xvec,
                      std::vector<double>& yvec) const;

  // Setter methods
  void setHFtableCoords(double x, double y, double v);
  void setHBHEtableCoords(double eta, double phi);
  void setChamberHits(char chamberch, const std::vector<double>& xvec, const std::vector<double>& yvec);

private:
  double hfTableX_, hfTableY_, hfTableV_;
  double hbheTableEta_, hbheTablePhi_;

  std::vector<double> ax_, ay_, bx_, by_, cx_, cy_, dx_, dy_, ex_, ey_, fx_, fy_, gx_, gy_, hx_, hy_;
};

std::ostream& operator<<(std::ostream& s, const HcalTBEventPosition& htbep);

#endif
