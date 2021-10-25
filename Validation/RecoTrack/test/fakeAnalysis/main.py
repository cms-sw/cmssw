#!/usr/bin/env python3

import ROOT
from array import array

from collections import OrderedDict
import pickle

from Validation.RecoTrack.python.plotting.ntuple import *

import analysis as an
import graphics as gr
#import graphics_vertical as gr # Use this to plot helixes better, this, however, supports only vertical z-axis and no filters except fake filter.


def main():
    '''
    The main function, below are commented templates of function calls to perform analysis.
    '''

    ntuple = TrackingNtuple("/afs/cern.ch/work/j/jmaanpaa/public/trackingNtuple_810pre6_v5.root")
    #ntuple = TrackingNtuple("trackingNtuple_pre9_1000ev.root") # Uncomment this for 1000 event data
    pl = gr.EventPlotter() 

    ###### USEFUL COMMANDS, UNCOMMENT TO USE ######

    PlotFakes(ntuple) # Plots the fakes in 3D one by one

    #Create_Histograms_MaxMatched_FakeTracks(ntuple)
    #Create_Histograms_MaxMatched_RealTracks(ntuple)

    #Create_Histograms_Classification(ntuple) # CREATES tHE MAIN CLASSIFICATION HISTOGRAMS
    #Create_Histograms_IncludedDecayHitFrac(ntuple)
    #Create_Histograms_UnmatchedInfo(ntuple)
    #Create_Histograms_PixInfo(ntuple)

    #Create_EndOfTrackingHistograms(ntuple)
    #Create_EndOfTrackingHistogramsReal(ntuple)
    #Create_EOT_DistanceHistogram(ntuple)

    #Create_ResolutionHistograms(ntuple)

    #print an.Calculate_AmountOfFakes(ntuple)


    ###### NOT THAT USEFUL COMMANDS, NOT TESTED AFTER UPDATES ######
     
    #ntuple_iter = iter(ntuple)#__iter__()
    #event = ntuple_iter.next()
    #tracks = event.tracks()

    #Create_PtHistograms(ntuple)
     
    #an.Save_Normalisation_Coefficients(ntuple)
    
    #Create_EndOfTrackingPointsPlot(ntuple)
    #an.EndOfTrackingDetectorInfo(Load_File("End_list_file_layermiss.dmp"),[0],True)
 
    #Create_Histograms_SharedHitsFracs_RealTracks(ntuple)
    #Create_Histograms_SharedHitsFracs(ntuple)  

    #PlotEvent3DHits(event, "PSG")
    #PlotXY(event, [-10,10])
    #PlotZY(event,flag="P")#, [-10,10])
    #PlotPixelGluedTracks3D(tracks)
    #PlotTracks3D(FindFakes(event))
    #PlotTracks3D(FindUntrackedParticles(event))
    #EfficiencyRate(ntuple, 100)
    #for track in an.FindFakes(event):
#	pl.DrawTrackTest(track)

    #pl.PlotEvent3DHits(event)

    '''
    track = iter(tracks).next()
    pl.Plot3DHits(track)
    pl.Draw()
    
    for track in tracks:
	if track.hits():
	    #pl.PlotTrack3D(track)
	    pl.Plot3DHelixes([track])
	    pl.Plot3DHits(track)
	    #pl.Plot3DHelixes(an.FindFakes(event))
	    pl.Draw()
    '''
    #pl.ParticleTest(event.trackingParticles(),True)
    #pl.PlotFakes(event,1,1,None)#[])
    #pl.PlotZY(event)
    #pl.Draw()


def Save_To_File(items, name):
    f = open(name,'w')
    pickle.dump(items,f)
    f.close()

def Load_File(name):
    f = open(name, 'r')
    items = pickle.load(f)
    f.close()
    return items

def PlotFakes(ntuple):
    ''' Plots fakes with the PlotFakes command in the graphics module. '''
    pl = gr.EventPlotter()
    for event in ntuple:
        # Fill the filters with corresponding class indexes or detector layers to filter everything else from the plotting
	pl.PlotFakes(event, True, fake_filter = [], end_filter = [], last_filter = "", particle_end_filter = "")

def Create_PtHistograms(ntuple):
    ''' Creates pt histograms for fake and true tracks. '''
    pl = gr.EventPlotter()
    
    fake_pts, true_pts = an.TrackPt(ntuple)
    fake_pts = [min(pt, 4.99) for pt in fake_pts]
    true_pts = [min(pt, 4.99) for pt in true_pts]
    pl.ClassHistogram(fake_pts, "Fake p_{t}", False, False)
    pl.ClassHistogram(true_pts, "True p_{t}", False, False)

def Create_ResolutionHistograms(ntuple):
    '''
    Performs resolution analysis and plots histograms from the results.
    Comment and uncomment lines depending on if you have saved the files.
    '''
    pl = gr.EventPlotter()

    res_data_fakes = an.Resolution_Analysis_Fakes(ntuple, 100)
    #Save_To_File(res_data_fakes, "resolutions_fake_matches1000ev.dmp")
    #res_data_fakes = Load_File("resolutions_fake_matches1000ev.dmp")

    #pl.ResolutionHistograms(res_data_fakes)
    #pl.HistogramSigmaPt(res_data_fakes)

    res_data_trues = an.Resolution_Analysis_Trues(ntuple, 100)
    #Save_To_File(res_data_trues, "resolutions_trues1000ev.dmp")
    #res_data_trues = Load_File("resolutions_trues1000ev.dmp")

    res_data_fakes_NLay = an.Resolution_Analysis_Fakes(ntuple, 100, real_criterion = ["NLay",6])
    #Save_To_File(res_data_fakes_NLay, "resolutions_fake_NLay_matches1000ev.dmp")
    #res_data_fakes_NLay = Load_File("resolutions_fake_NLay_matches1000ev.dmp")

    #pl.ResolutionHistograms(res_data_fakes)
    #pl.HistogramSigmaPt(res_data_fakes)

    #pl.ResolutionHistograms(res_data_trues)
    #pl.HistogramSigmaPt(res_data_trues)
    
    #pl.ResolutionHistogramsCombine(res_data_trues, res_data_fakes)
    pl.ResolutionHistogramsNormalised(res_data_trues, res_data_fakes)
    pl.ResolutionHistogramsNormalisedCumulative(res_data_trues, res_data_fakes)

    #pl.HistogramSigmaPtCombined(res_data_trues, res_data_fakes)
    #pl.HistogramSigmaPtDual(res_data_trues, res_data_fakes)
    pl.ResolutionsSigmaVsPt(res_data_trues, res_data_fakes)
    
    pl.ResolutionHistogramsNormalised(res_data_trues, res_data_fakes_NLay)
    pl.ResolutionHistogramsNormalisedCumulative(res_data_trues, res_data_fakes_NLay)

    pl.ResolutionsSigmaVsPt(res_data_trues, res_data_fakes_NLay)



def Create_EndOfTrackingHistograms(ntuple):
    '''
    Creates various histograms related to end of tracking
    '''
    pl = gr.EventPlotter()

    ''' Comment/uncomment these lines depending on if you have saved the end of tracking file'''
    end_list = an.Get_EndOfTrackingPoints(ntuple, 100, [])
    #Save_To_File(end_list,"End_list_file_layermiss_invalid4_fix_4class.dmp")

    #end_list = Load_File("End_list_file_layermiss_invalid4_fix_4class.dmp") #an.Get_EndOfTrackingPoints(ntuple, 10, [])

    #end_list = an.Get_EndOfTrackingPointsReal(ntuple, 100)
    #Save_To_File(end_list,"End_list_file_real_invalid.dmp")

    #end_list = Load_File("End_list_file_real_invalid.dmp")
    '''
    for end in end_list:
	if end.end_class == 0 and end.last_str == end.particle_end_str and end.last_str == "BPix3":
	    print end.last
	    print end.particle_end
	    print end.fake_end
	    print end.last_str, end.particle_end_str, end.fake_end_str
	    print end.end_class
	    print "*****"
    '''
    an.Analyse_EOT_ParticleDoubleHit(end_list)
    pl.EndOfTrackingHistogram(end_list, "end_class",[],[], False, False) #last false will adjust the bpix3 filter
    #pl.EndOfTrackingHistogram(end_list, "invalid_reason_last",[],[],False, True)
    pl.EndOfTrackingHistogram(end_list, "invalid_reason_fake",[0],[],False, False)
    pl.EndOfTrackingHistogram(end_list, "particle_pdgId",[0],[],False, False)
    #pl.EndOfTrackingHistogram(end_list, "invalid_reason_particle",[],[],False, False)
    #pl.EndOfTrackingHistogram(end_list, "particle_end_layer",[0],[],False, True)

    pl.EndOfTrackingHistogram(end_list, "last_layer",[0],[],False, False)# 11,13,21
    pl.EndOfTrackingHistogram(end_list, "fake_end_layer",[0],[],False, False)
    pl.EndOfTrackingHistogram(end_list, "particle_end_layer",[0],[],False, False)

    errors = an.Analyse_EOT_Error(end_list, True, "same", False)  # Same detectior id
    pl.ClassHistogram(errors,"Distance between EOT fake and particle hit", False, False)
    errors = an.Analyse_EOT_Error(end_list, True, "different", False)  # Different detectior id
    pl.ClassHistogram(errors,"Distance between EOT fake and particle hit", False, False)
    an.EndOfTrackingDetectorInfo(end_list,[0],True)


def Create_EndOfTrackingHistogramsReal(ntuple):
    '''
    Create end of tracking histograms for true tracks.
    '''
    pl = gr.EventPlotter()

    ''' Comment/uncomment these lines depending on if you have saved the end of tracking file'''
    end_list = an.Get_EndOfTrackingPointsReal(ntuple, 100)
    #Save_To_File(end_list,"End_list_file_real_fixed.dmp")

    #end_list = Load_File("End_list_file_real_fixed.dmp") #an.Get_EndOfTrackingPoints(ntuple, 10, [])

    '''
    for end in end_list:
	if end.end_class == 0 and end.last_str == end.particle_end_str and end.last_str == "BPix3":
	    print end.last
	    print end.particle_end
	    print end.fake_end
	    print end.last_str, end.particle_end_str, end.fake_end_str
	    print end.end_class
	    print "*****"
    '''
    pl.EndOfTrackingHistogram(end_list, "end_class",[],[])
    pl.EndOfTrackingHistogram(end_list, "last_layer",[0],[],False)
    pl.EndOfTrackingHistogram(end_list, "fake_end_layer",[0],[],False)
    pl.EndOfTrackingHistogram(end_list, "particle_end_layer",[0],[],False)

def Create_EOT_DistanceHistogram(ntuple):
    pl = gr.EventPlotter()
    end_list = an.Get_EndOfTrackingPoints(ntuple, 100, [])
    #Save_To_File(end_list,"End_list_file_layermiss.dmp")
    #end_list = Load_File("End_list_file_layermiss.dmp")

    errors = an.Analyse_EOT_Error(end_list, True)
    
    pl.ClassHistogram(errors,"Distance between EOT fake and particle hit", False, False)
    
def Create_EndOfTrackingPointsPlot(ntuple):
    # NOT TESTED AFTER UPDATE
    pl = gr.EventPlotter()

    end_list = an.Get_EndOfTrackingPoints(ntuple, 100, [])
    last = [], fake_ends = [], particle_ends = [], end_classes = [], fake_classes = []
    for end in end_list:
	last.append(end.last)
	fake_ends.append(end.fake_end)
	particle_ends.append(end.particle_end)
	end_classes.append(end.end_class)
	fake_classes.append(end.fake_class)

    '''
    pl.PlotEndOfTrackingPoints3D(last,[])#, fail)
    pl.PlotEndOfTrackingPoints3D(fake_ends,[],[],2)
    pl.PlotEndOfTrackingPoints3D(particle_ends,[],[],3)
    #pl.PlotDetectorRange()
    pl.Draw()
    '''
    pl.PlotEndOfTracking3D(last, fake_ends, particle_ends, end_classes, fake_classes)
    pl.PlotDetectorRange()
    pl.Draw()

    pl.Reset()
    pl.PlotEndOfTracking3D(last, fake_ends, particle_ends, end_classes, fake_classes, [0],[])
    pl.PlotDetectorRange()
    pl.Draw()

    pl.Reset()
    pl.PlotEndOfTracking3D(last, fake_ends, particle_ends, end_classes, fake_classes, [],[10])
    pl.PlotDetectorRange()
    pl.Draw()

    pl.Reset()
    pl.PlotEndOfTracking3D(last, fake_ends, particle_ends, end_classes, fake_classes, [],[13])
    pl.PlotDetectorRange()
    pl.Draw()

def Create_Histograms_IncludedDecayHitFrac(ntuple):
    '''
    Creates plots related to the "multiple matches including a decay interaction" -class.
    '''
    pl = gr.EventPlotter()
    
    daughter_frac, parent_frac, total_frac = an.Calculate_IndludedDecayHitFractions(ntuple, 100)
    pl.ClassHistogram(daughter_frac, "Daughter share of hits from fake", False, False)
    pl.ClassHistogram(parent_frac, "Parent share of hits from fake", False, False)
    pl.ClassHistogram(total_frac, "Total share of hits from fake", False, False)


def Create_Histograms_UnmatchedInfo(ntuple):
    pl = gr.EventPlotter()
    
    results = an.Calculate_UnmatchedSharedHits(ntuple, 100)
    pl.ClassHistogram(results, "Maximum number of shared hits in unmatched and unclassified classes", False, False)

def Create_Histograms_SharedHitsFracs(ntuple):
    pl = gr.EventPlotter()
    
    hits_shared, hit_fractions = an.Calculate_SharedHits(ntuple, 100, []) 
    pl.ClassHistogram(hits_shared, "Maximum number of shared hits in all classes", False, False)
    pl.ClassHistogram(hit_fractions, "Shared hit fractions in all classes", False, False)

    hits_shared, hit_fractions = an.Calculate_SharedHits(ntuple, 100, [10,11,12,13]) 
    pl.ClassHistogram(hits_shared, "Maximum number of shared hits in single match classes", False, False)
    pl.ClassHistogram(hit_fractions, "Shared hit fractions in single match classes", False, False)

    hits_shared, hit_fractions = an.Calculate_SharedHits(ntuple, 100, [20,21,22,23]) 
    pl.ClassHistogram(hits_shared, "Maximum number of shared hits in multiple match classes", False, False)
    pl.ClassHistogram(hit_fractions, "Shared hit fractions in multiple match classes", False, False)

    hits_shared, hit_fractions = an.Calculate_SharedHits(ntuple, 100, [10]) 
    pl.ClassHistogram(hits_shared, "Maximum number of shared hits in other single match class", False, False)
    pl.ClassHistogram(hit_fractions, "Shared hit fractions in other single match class", False, False)

    hits_shared, hit_fractions = an.Calculate_SharedHits(ntuple, 100, [13]) 
    pl.ClassHistogram(hits_shared, "Maximum number of shared hits in single decay match class", False, False)
    pl.ClassHistogram(hit_fractions, "Shared hit fractions in single decay match class", False, False)

def Create_Histograms_SharedHitsFracs_RealTracks(ntuple):
    pl = gr.EventPlotter()
    
    hits_shared, hit_fractions = an.Calculate_SharedHits_RealTracks(ntuple, 100) 
    pl.ClassHistogram(hits_shared, "Maximum number of shared hits in real reconstructed tracks", False, False)
    pl.ClassHistogram(hit_fractions, "Shared hit fractions in real reconstructed tracks", False, False)

def Create_Histograms_MaxMatched_FakeTracks(ntuple):
    pl = gr.EventPlotter()
    #results = an.Calculate_MaxMatchedHits(ntuple, 100)
    #pl.ClassHistogram(results, "Max Matched Hits", False, False) 
     
    results = an.Calculate_MaxMatchedConsecutiveHits(ntuple, 100) 
    pl.ClassHistogram(results, "Matched consecutive hits of fake", False, False, True, 0, 20, 20, True)

    results = an.Calculate_MaxMatchedConsecutiveHits(ntuple, 100, True) 
    pl.ClassHistogram(results, "Consecutive hit fraction of fake", False, False, False, 0, 1.1, 11, True)

def Create_Histograms_MaxMatched_RealTracks(ntuple):
    pl = gr.EventPlotter()
    results = an.Calculate_MaxMatchedHits_RealTracks(ntuple, 100)
    pl.ClassHistogram(results, "Matched Hits of real reconstructed tracks", False, False) 
    
    results = an.Calculate_MaxMatchedConsecutiveHits_RealTracks(ntuple, 100)
    pl.ClassHistogram(results, "Matched consecutive hits of real reconstructed track", False, False)

def Create_Histograms_PixInfo(ntuple):
    pl = gr.EventPlotter()
    
    nPix_data, nLay_data = an.Calculate_MatchPixelHits(ntuple, 100)
    pl.ClassHistogram(nPix_data, "Number of pixelhits in matches", False, False)
    pl.ClassHistogram(nLay_data, "Number of pixel layers in matches", False, False)

def Create_Histograms_Classification(ntuple):
    '''
    Creates histograms from classification of fakes.
    '''
    pl = gr.EventPlotter()
    '''
    results = an.ClassifyEventFakes(ntuple, 10, False)
    pl.ClassHistogram(results, "Classification of all fakes", an.classes, True) 
    
    results = an.Calculate_MaxMatchedConsecutiveHits(ntuple, 100)
    pl.ClassHistogram(results, "Max matched consecutive hits", False, False) 
    
    results = an.Calculate_MaxMatchedHits(ntuple, 100)
    pl.ClassHistogram(results, "Max Matched Hits", False, False) 
     
    results = an.Calculate_MaxMatchedConsecutiveHits(ntuple, 100, True)
    pl.ClassHistogram(results, "Hit fraction of fake", False, False) #"Max matched consecutive hits with respect to fake", False, False)
    '''
 
    results = an.ClassifyEventFakes(ntuple, 100, False) # ORIGINAL MATCH CRITERION
    #results = an.ClassifyEventFakes(ntuple, 100, False, real_criterion = ["NLay", 6]) # SHARED LAYERS (NLay) MATCH CRITERION

    #pl.ClassHistogram(results, "Classification of all fakes", an.classes, True) 

    all_classified = OrderedDict()
    all_classified["No matches"] = results[0]
    all_classified["One match"] = sum([results[i] for i in [10,11,12,13]])
    all_classified["Multiple matches"] = sum([results[i] for i in [20,21,22,23]])
    all_classified["Loose match"] = results[-1]

    pl.ClassHistogram(all_classified, "Classification of all fakes", False, True)

    single_classified = OrderedDict()
    single_classified["Decayed and reconstructed"] = results[11]
    single_classified["Only reconstructed"] = results[12]
    single_classified["Only decayed"] = results[13]
    single_classified["Neither"] = results[10]
    pl.ClassHistogram(single_classified, "Classification of fakes with single match", False, True)

    multi_classified = OrderedDict()
    multi_classified["Includes a decay interaction"] = results[21]
    multi_classified["All matches reconstructed"] = results[22]
    multi_classified["Some matches reconstructed"] = results[23]
    multi_classified["Other"] = results[20]
    pl.ClassHistogram(multi_classified, "Classification of fakes with multiple matches", False, True)



if __name__ == "__main__":
    main()

