#ifndef SimG4Core_LumiMonitorFilter_H
#define SimG4Core_LumiMonitorFilter_H

class GenParticle;

class LumiMonitorFilter
{
public:
  LumiMonitorFilter();
  virtual ~LumiMonitorFilter();

  void Describe() const; 
  bool isGoodForLumiMonitor(const GenParticle*) const;

private:

};

#endif
