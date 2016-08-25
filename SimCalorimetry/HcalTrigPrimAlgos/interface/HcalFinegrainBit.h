#ifndef HcalSimAlgos_HcalFinegrainBit_h
#define HcalSimAlgos_HcalFinegrainBit_h

#include <array>
#include <bitset>

class HcalFinegrainBit {
   public:
      typedef std::array<std::bitset<7>, 2> Tower;

      HcalFinegrainBit(int version) : version_(version) {};

      std::bitset<4> compute(const Tower&) const;
   private:
      int version_;
};

#endif
