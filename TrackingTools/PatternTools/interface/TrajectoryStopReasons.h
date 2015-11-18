#ifndef TRAJECTORYSTOPREASONS_H
#define TRAJECTORYSTOPREASONS_H

enum class StopReason {
  MAX_HITS = 0,
  MAX_LOST_HITS = 1,
  MAX_CONSECUTIVE_LOST_HITS = 2,
  LOST_HIT_FRACTION = 3,
  MIN_PT = 4,
  CHARGE_SIGNIFICANCE = 5,
  LOOPER = 6,
  UNKNOWN=255 // this is the max allowed since it will be streames as type uint8_t
};

#endif
