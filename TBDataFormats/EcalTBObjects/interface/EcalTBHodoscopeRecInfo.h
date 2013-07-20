#ifndef RECECAL_ECALTBHODOSCOPERECINFO_H
#define RECECAL_ECALTBHODOSCOPERECINFO_H 1

#include <ostream>

/** \class EcalTBHodoscopeRecInfo
 *  Simple container for Hodoscope reconstructed informations 
 *
 *
 *  $Id: EcalTBHodoscopeRecInfo.h,v 1.1 2006/04/21 09:26:34 meridian Exp $
 */


class EcalTBHodoscopeRecInfo {
 public:
  EcalTBHodoscopeRecInfo() {};
  EcalTBHodoscopeRecInfo(const float& xpos, const float& ypos, const float& xslope, const float& yslope, const float& xqual, const float& yqual): pos_x_(xpos), pos_y_(ypos), slope_x_(xslope), slope_y_(yslope), qual_x_(xqual), qual_y_(yqual)
    {
    };

  ~EcalTBHodoscopeRecInfo() {};
  
  float posX() const { return pos_x_; }
  float posY() const { return pos_y_; }

  float slopeX() const { return slope_x_; }
  float slopeY() const { return slope_y_; }

  float qualX() const { return qual_x_; }
  float qualY() const { return qual_y_; }

  void setPosX(const float& xpos) { pos_x_=xpos; }
  void setPosY(const float& ypos) { pos_y_=ypos; }
  
  void setSlopeX(const float& xslope) { slope_x_=xslope; }
  void setSlopeY(const float& yslope) { slope_y_=yslope; }
  
  void setQualX(const float& xqual) { qual_x_=xqual; }
  void setQualY(const float& yqual) { qual_y_=yqual; }

 private:

  float pos_x_;
  float pos_y_;

  float slope_x_;
  float slope_y_;

  float qual_x_;
  float qual_y_;
  
  
};

std::ostream& operator<<(std::ostream&, const EcalTBHodoscopeRecInfo&);
  
#endif
