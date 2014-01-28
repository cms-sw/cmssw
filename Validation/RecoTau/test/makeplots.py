#!/usr/bin/env python

import re
import pprint
import copy
import os
import sys

# Import validation configuration
import Validation.RecoTau.steering as steering
from RecoLuminosity.LumiDB import argparse

_global_tcolor_counter = 0
def build_color(hex_str):
    return hex_str
    red = int(hex_str[1:3], 16)/255.0
    green = int(hex_str[3:5], 16)/255.0
    blue = int(hex_str[5:7], 16)/255.0
    #output = ROOT.TColor(270, red, green, blue)
    output = ROOT.TColor(_global_tcolor_counter)
    _global_tcolor_counter += 1
    return output

def get_objects(directory, type=None):
    ''' Get all of objects of a given type from a directory '''
    keys = directory.GetListOfKeys()
    for key in keys:
        obj = key.ReadObj()
        if type is None or obj.IsA().InheritsFrom(type):
            yield obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Make RecoTauTag efficiency plots"
    )
    parser.add_argument('-i', metavar='file', help='Input file')
    parser.add_argument('-o',  default='plots', help='Output directory')
    parser.add_argument('-truth', default=False, action='store_true',
                        help="Use MC for denominator")
    parser.add_argument('-log', default=False, action='store_true')

    options=parser.parse_args()

    # Work around
    #class Empty(object):
        #pass
    #options = Empty()
    #import sys
    #options.i = "val_qcd.root"
    #options.o = "plots_qcd"
    #options.truth = True
    #options.log = True

    sys.argv = []
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gROOT.SetStyle("Plain")

    output_dir = options.o
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir):
        print "Output path:", options.o, \
                " is not a directory and cannot be created!"
        sys.exit(1)

    # Open file
    input_file = ROOT.TFile(options.i, "READ")
    # Get the list of all directories
    directory_matcher = re.compile(
        r"(?P<algorithm>\w+?)(?P<isGenMatched>GenMatched)*"
        r"Select(?P<discriminator>\w+)Plots")
    # Now update our steering dictionary
    for subdirectory in get_objects(input_file, "TDirectory"):
        print "Found directory", subdirectory.GetName()
        match = directory_matcher.match(subdirectory.GetName())
        if match:
            # Get the infromation about this algorithm
            algo_info = steering.algorithms[match.group('algorithm')]
            # If it is matched, make the matched version a new algorithm- a
            # clone of the old one.
            is_matched = match.group('isGenMatched') is not None
            if is_matched:
                new_producer = match.group('algorithm') + "GenMatched"
                if new_producer not in steering.algorithms:
                    steering.algorithms[new_producer] = copy.copy(algo_info)
                    # Delete any previously made discriminator mappings as they
                    # correspond to the non-genmatched producer
                    if 'disc_map' in algo_info:
                        del steering.algorithms[new_producer]['disc_map']
                # Use the new matched version
                algo_info = steering.algorithms[new_producer]
                algo_info['matched'] = True
            else:
                algo_info['matched'] = False
            # Store a link to the plot directory in our steering
            # The true or false indicates whether or not it is matched to the
            # MC truth
            algo_disc_map = algo_info.setdefault('disc_map', {})
            # Keep track of the discriminator folder
            algo_disc_map[match.group('discriminator')] = subdirectory
    print "Directories built"

    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(steering.algorithms)

    # Get the truth plots, if they exist
    steering.algorithms['truth_plots'] = input_file.Get('plotTruth')

    if options.truth and not steering.algorithms['truth_plots']:
        print "Requested to use [-truth] option - but the truth plots are "\
        "not found in the input file!"
        sys.exit(1)

    # Define the variables to plot efficiency versus:
    eff_vars = ['pt', 'eta', 'phi']
    eff_vars_nicenames = ['P_{T}', '#eta', '#phi']
    eff_var_names = dict(zip(eff_vars, eff_vars_nicenames))

    # Determine how to translate the variable depending on in the variable is
    # matched.
    var_getters = {
        False : lambda var: var,
        True : lambda var: var + "OfMatched"
    }

    print "Computing efficiencies"
    # Plot each var
    for var in eff_vars:
        # For each algorithm
        for algorithm in steering.algorithms.keys():
            # Skip the truth plots folder
            if algorithm == 'truth_plots':
                continue
            algo_info = steering.algorithms[algorithm]
            # Get the denominator - different whether we are using truth or not
            denominator_histo = None
            if algo_info['matched']:
                denominator_histo = steering.algorithms['truth_plots'].Get(var)
            else:
                denominator_histo = algo_info['disc_map']['noSelection'].Get(var)
            denominator_histo = denominator_histo.Clone("denom_redo")
            #denominator_histo.Rebin(2)

            # Build efficiency map - each numerator divided by our denominator
            eff_map = algo_info.setdefault('eff_map', {})
            eff_map_var = eff_map.setdefault(var, {})

            # Build an efficiency for each discriminator.
            for discriminator, folder in algo_info['disc_map'].iteritems():
                print "Computing efficiency:", algorithm, ":", discriminator
                # For the numerator, we need to map the var name (pt etc) to
                # the one that corresponds to the matched pt.
                numerator_histo = folder.Get(
                    var_getters[algo_info['matched']](var))
                if not numerator_histo:
                    print "Can't build efficiency for", discriminator,\
                            "can't find numerator! Matched:", \
                            algo_info['matched'], "This shouldn't happen."
                    continue
                #numerator_histo.Rebin(2)
                # Build efficiency
                efficiency = ROOT.TGraphAsymmErrors(
                    numerator_histo, denominator_histo)
                eff_map_var[discriminator] = efficiency
                # Set marker color & size
                efficiency.SetMarkerColor(
                    build_color(algo_info['color']))
                efficiency.SetMarkerSize(1)
                efficiency.SetMarkerStyle(20)

    workaround_canvas = ROOT.TCanvas("blah", "blah", 800, 1200)
    workaround_canvas.Divide(1, 2)
    workaround_canvas.cd(1)
    canvas = ROOT.gPad
    if options.log:
        canvas.SetLogy(True)

    # Okay, now all the efficiencies have been computed.  We can make some plots
    # First just plot every efficiency
    for var in eff_vars:
        for algorithm, algo_info in steering.algorithms.iteritems():
            if algorithm == 'truth_plots':
                continue
            for disc, tgraph in algo_info['eff_map'][var].iteritems():
                tgraph.Draw("ap")
                tgraph.GetHistogram().SetTitle(
                    algo_info['nicename'] + " " +
                    steering.discriminator_nice_name(disc))
                tgraph.GetHistogram().GetXaxis().SetTitle(eff_var_names[var])
                canvas.SaveAs(os.path.join(
                    output_dir,
                    "_".join([algorithm, disc, var]) + ".pdf")
                )

        # Now plot all our comparison plots
        for comparison, comparison_info in steering.comparisons.iteritems():
            print "Building comparison plot", comparison
            #continue
            comparison_info['legend'] = ROOT.TLegend(0.1, 0.6, 0.5, 0.9)
            comparison_info['legend'].SetFillStyle(0)
            comparison_info['legend'].SetBorderSize(0)
            comparison_info['background'] = None
            for algo_source, discriminator in comparison_info['plots']:
                print " --- adding %s:%s" % (algo_source, discriminator)
                efficiency = steering.algorithms[algo_source]['eff_map'][var][
                    discriminator]
                # Build legend
                comparison_info['legend'].AddEntry(
                    efficiency,
                    algo_source + ":" + steering.discriminator_nice_name(discriminator), "p")
                if not comparison_info['background']:
                    efficiency.Draw("ape")
                    comparison_info['background'] = efficiency.GetHistogram()
                    comparison_info['background'].SetMinimum(5e-4)
                    comparison_info['background'].SetMaximum(1.0)
                    comparison_info['background'].GetXaxis().SetTitle(
                        eff_var_names[var])
                    #efficiency.Draw("pe")
                else:
                    efficiency.Draw("pe, same")
            comparison_info['legend'].Draw()
            canvas.SaveAs(os.path.join(
                output_dir, "_".join([comparison, var])) + ".pdf")
