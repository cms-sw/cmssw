import FWCore.ParameterSet.Config as cms

from TrackingTools.Producers.smartPropagatorESProducer_cfi import smartPropagatorESProducer as _smartPropagatorESProducer
SmartPropagator = _smartPropagatorESProducer.clone()
