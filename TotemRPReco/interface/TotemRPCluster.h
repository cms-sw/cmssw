/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_TotemRPReco_TotemRPCluster
#define DataFormats_TotemRPReco_TotemRPCluster

/**
 *\brief Cluster of TOTEM RP strip hits.
 **/
class TotemRPCluster
{
 public:
  TotemRPCluster(unsigned int det_id, unsigned short str_beg, unsigned short str_end)
    : det_id_(det_id), str_beg_(str_beg), str_end_(str_end)
    {}

  TotemRPCluster() : det_id_(0), str_beg_(0), str_end_(0) {}

  inline void DetId(unsigned int det_id) {det_id_ = det_id;}
  inline unsigned int DetId() const {return det_id_;}

  inline void StrBeg(unsigned short str_beg) {str_beg_ = str_beg;}
  inline unsigned short StrBeg() const {return str_beg_;}

  inline void StrEnd(unsigned short str_end) {str_end_ = str_end;}
  inline unsigned short StrEnd() const {return str_end_;}

  inline int GetNumberOfStrips() const {return (str_end_-str_beg_+1);}

  inline double CentreStripPos() const {return (str_beg_+str_end_)/2.0;}

 private:
  unsigned int det_id_;
  unsigned short str_beg_;
  unsigned short str_end_;
};



inline bool operator<( const TotemRPCluster& one, const TotemRPCluster& other)
{
  if(one.DetId() < other.DetId())
    return true;
  else if(one.DetId() == other.DetId())
    return one.CentreStripPos() < other.CentreStripPos();
  else 
    return false;
}


#endif
