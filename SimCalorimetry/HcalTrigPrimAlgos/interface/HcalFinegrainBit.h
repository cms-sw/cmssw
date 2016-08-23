#ifndef HcalSimAlgos_HcalFinegrainBit_h
#define HcalSimAlgos_HcalFinegrainBit_h

#include <array>
#include <bitset>

class HcalFinegrainBit {
   public:
      typedef std::array<std::bitset<7>, 2> Tower;

      HcalFinegrainBit(int version) : version_(version) {};

      int compute(const Tower&) const;
   private:
      int version_;
};

#endif
