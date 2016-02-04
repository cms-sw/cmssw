#ifndef DIGIECAL_ECALTBHODOSCOPEPLANERAWHITS_H
#define DIGIECAL_ECALTBHODOSCOPEPLANERAWHITS_H 1


#include <ostream>
#include <vector>
#include <boost/cstdint.hpp>

/** \class EcalTBHodoscopePlaneRawHits
 *  Simple container for rawHits 
 *
 *
 *  $Id: EcalTBHodoscopePlaneRawHits.h,v 1.3 2006/06/06 15:37:00 meridian Exp $
 */

class EcalTBHodoscopePlaneRawHits {
 public:
  EcalTBHodoscopePlaneRawHits(): rawChannelHits_(0) {};

  EcalTBHodoscopePlaneRawHits(unsigned int channels) 
    {
      rawChannelHits_.reserve(channels);
      for (unsigned int i=0;i<channels;i++)
	rawChannelHits_[i]=0;
    }
  
  /// Get Methods
  unsigned int channels() const { return rawChannelHits_.size(); } 
  const std::vector<bool>& hits() const { return rawChannelHits_; }

  unsigned int numberOfFiredHits() const 
    {
      int numberOfHits=0;
      for (unsigned int i=0;i<rawChannelHits_.size();i++)
	if (rawChannelHits_[i])
	  numberOfHits++;
      return numberOfHits;
    }

  bool operator[](unsigned int i) const { return rawChannelHits_[i]; }
  bool isChannelFired(unsigned int i) const { return rawChannelHits_[i]; }

  /// Set methods
  void setChannels(unsigned int size)
    {
      rawChannelHits_.resize(size);
    };
  
  void addHit(unsigned int i) 
    {
      if (rawChannelHits_.size() < i+1 )
	rawChannelHits_.resize(i+1);
      rawChannelHits_[i]=true; 
    };

  void setHit(unsigned int i,bool status) 
    {
      if (rawChannelHits_.size() < i+1 )
	rawChannelHits_.resize(i+1);
      rawChannelHits_[i]=status; 
    };


 private:
  std::vector<bool> rawChannelHits_;
  
};

std::ostream& operator<<(std::ostream&, const EcalTBHodoscopePlaneRawHits&);
  
#endif
