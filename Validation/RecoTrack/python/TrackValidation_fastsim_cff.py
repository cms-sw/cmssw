import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackValidation_cff import *
TrackAssociatorByHits.associateStrip = False
TrackAssociatorByHits.associatePixel = False
TrackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']
multiTrackValidator.UseAssociators = True
###must be commented in normal running
###multiTrackValidator.outputFile='validationPlots.root'


