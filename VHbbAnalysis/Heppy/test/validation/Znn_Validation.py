# Sean-Jiun Wang <swang373@phys.ufl.edu>
# Proverbs 22:29

import argparse
import ROOT

"""
A script for validating different ntuple versions in the context of ZnnHbb. 

To adapt this to a different channel, simply change the outdir (line 115)
and validation_vars (line 128) variables to suit your needs.
"""

#------------------
# Auxiliary Program
#------------------

def make_validation_plot(var, old_tree, new_tree, cuts, outdir):
    
    """
    For the variable, print out the mean and RMS of
    its distribution and save an overlaid histogram.
    
    Parameters
    ----------
    var      : list, [string, int, int, int]
               The first item specifies the validation variable name. 
               The remaining items specify the number of bins along the x-axis, 
               the x-axis lower bound, and the x-axis upper bound, respectively.
    old_tree : ROOT.TTree
               The tree from the old ntuple.
    new_tree : ROOT.TTree
               The tree from the new ntuple.
    cuts     : string
               Any selections to be applied to both trees.
    outdir   : string
               The name of the directory where the histograms are saved.
    """

    # Draw the validation variable histograms.
    canvas = ROOT.TCanvas('canvas', '', 500, 500)

    old_tree.Draw('%s>>Old(%s,%s,%s)' % (var[0], var[1], var[2], var[3]), cuts)
    new_tree.Draw('%s>>New(%s,%s,%s)' % (var[0], var[1], var[2], var[3]), cuts)

    hists = [ROOT.gDirectory.Get('Old'), ROOT.gDirectory.Get('New')]

    # Print out the mean and RMS of the distributions.
    mean = [hist.GetMean() for hist in hists]
    RMS =  [hist.GetRMS() for hist in hists]

    print '\n-------------------------------------------------------'
    print var[0].center(15,' ') + '    Mean            :   RMS'
    print '-------------------------------------------------------'
    print 'OldNtuple'.center(15,' ') + ':   ' + str(mean[0]).ljust(16,' ') + ':   ' + str(RMS[0]).ljust(15,' ')
    print 'NewNtuple'.center(15,' ') + ':   ' + str(mean[1]).ljust(16,' ') + ':   ' + str(RMS[1]).ljust(15,' ')
    print '-------------------------------------------------------'

    # Format line styles and normalize the histograms.
    for hist, color, in zip(hists, [ROOT.kBlack, ROOT.kRed]):
        hist.SetLineWidth(2)
        hist.SetLineColor(color)
        hist.Scale(1.0 / hist.Integral())
    
    # Overlay the histograms and rescale the y-axis for viewing.
    h_stack = ROOT.THStack('h_stack', '')
    for hist in hists:
        h_stack.Add(hist)
    h_stack.Draw('hist nostack')
    h_stack.GetXaxis().SetTitle(var[0])
    h_stack.SetMaximum(ROOT.gPad.GetUymax() * 1.35)

    # Format and draw a legend.
    legend = ROOT.TLegend(0.75, 0.75, 0.89, 0.89)
    legend.SetLineColor(0)
    for hist in hists:
        legend.AddEntry( hist, hist.GetName(), 'l')
    legend.Draw('same')

    # Format and draw LaTeX annotation.
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(12)
    latex.SetTextFont(62)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.15, 0.85, 'Znn Validation')

    # Save the canvas.
    canvas.SaveAs(outdir + var[0] + '.pdf')

    # Delete objects.
    canvas.IsA().Destructor(canvas)
    for hist in hists:
        hist.IsA().Destructor(hist)

#-------------
# Main Program
#-------------

if __name__ == '__main__':

    # Command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('OldNtuplePath', 
        help = 'The absolute path of the Ntuple generated using a previous Heppy version.')
    parser.add_argument('NewNtuplePath', 
        help = 'The absolute path of the Ntuple generated using the current Heppy version.')
    args = parser.parse_args()

    # Access the ntuples.
    OldNtuple = ROOT.TFile(args.OldNtuplePath, 'READ')
    NewNtuple = ROOT.TFile(args.NewNtuplePath, 'READ')

    # Create output directory.
    outdir = 'Znn_Validation_Plots/'

    if (ROOT.gSystem.AccessPathName(outdir)):
        ROOT.gSystem.mkdir(outdir)

    # Set ROOT to batch mode and remove the statistics box and title.
    ROOT.gROOT.SetBatch(1)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo + 1
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    # List of Validation Variables
    # Each variable has its own list specifying its name, number of x bins,
    # x-axis lower bound, and x-axis upper bound, respectively.
    validation_vars = [ 
                        ['HCSV_mass', 20, 0, 400],
                        ['HCSV_pt', 20, 0, 400], 
                        ['met_pt', 20, 0, 400],
                        ['Jet_btagCSV[0]', 20, 0, 1],
                        ['nJet', 16, 0, 16] 
                      ]

    for var in validation_vars:
        make_validation_plot(var, OldNtuple.tree, NewNtuple.tree, '', outdir)

    print "\nJob's done!"
