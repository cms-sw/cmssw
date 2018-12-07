import FWCore.ParameterSet.Config as cms

from SimFastTiming.FastTimingCommon.fastTimeDigitizer_cfi import *

from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import *
from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import _barrel_bar_MTDDigitizer

from Configuration.Eras.Modifier_phase2_timing_layer_bar_cff import phase2_timing_layer_bar
phase2_timing_layer_bar.toModify(mtdDigitizer, barrelDigitizer = _barrel_bar_MTDDigitizer.clone())
