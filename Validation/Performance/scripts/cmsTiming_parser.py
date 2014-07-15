#! /usr/bin/env python

import sys, os
import time
import optparse
from math import sqrt, log10, floor

import ROOT
    
def manipulate_log(outdir, logfile_name, secsperbin):
    """ Parses logfile_name and create an html report
        with information and plots, at the outdir path.
        Graphs' production is also done here.
    """
    ################################
    #### Parse of the log file. ####
    ################################ 
    logfile = open(logfile_name,'r')
    logfile_lines = iter(logfile.readlines())
    logfile.close()
    data = {}
    report = ''
    for line in logfile_lines:
        # Retrieve cpu time of every event.
        if 'TimeEvent>' in line:
            line = line[:-1] #no \n!
            content = line.split(' ')
            event =  int(content[1])
            seconds = float(content[3])
            data[event] = seconds
        # Fill the time report.
        elif 'TimeReport' in line:
            if '[sec]' in line:
                report += line.replace('TimeReport', '\n') 
            elif 'headings' in line:
                continue
            elif 'complete' in line:
                report += line.replace('TimeReport', '\n')
                for count in range(12):
                    line = next(logfile_lines)
                    report += line
                break
            else:
                report += line.replace('TimeReport', '')

    ##############################
    #### Graph and Histogram  ####
    ##############################
    __argv = sys.argv # trick for a strange behaviour of the TApp..
    sys.argv = sys.argv[:1]
    ROOT.gROOT.SetStyle("Plain") # style paranoia
    sys.argv = __argv
    #Cannot use this option when the logfile includes
    #a large number of events... PyRoot seg-faults.
    #Set ROOT in batch mode to avoid canvases popping up!
    ROOT.gROOT.SetBatch(1)

    # Save in file
    rootfilename = '%s/graphs.root' %outdir
    myfile = ROOT.TFile(rootfilename,'RECREATE')   
        
    # Set limits
    min_val = data[min(data, key=data.get)]
    max_val = data[max(data, key=data.get)]
    interval = max_val-min_val
    min_val = min_val - (interval*0.2)
    max_val = max_val + (interval*0.2)
    interval = max_val - min_val
    nbins = int(interval/secsperbin)

    # Initialize Histogram
    histo = ROOT.TH1F('Seconds per event','Seconds per event', nbins, min_val, max_val)
    histo.GetXaxis().SetTitle("s")    
    # Initialize Graph
    npoints = len(data)
    graph = ROOT.TGraph(npoints)
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)
    graph.SetLineColor(2)        
    graph.SetTitle('Seconds per event')
    graph.SetName('SecondsPerEvent')    
    graph.GetXaxis().SetTitle("Event")
    last_event = max(data)
    graph.GetXaxis().SetLimits(0, last_event)
    graph.GetYaxis().SetTitleOffset(1.3)
    graph.GetYaxis().SetTitle("s")
    graph.GetYaxis().SetRangeUser(0, max_val)
    # Fill them
    total_time = 0
    for event_num in data.keys():
        seconds = data[event_num]
        graph.SetPoint(event_num-1, event_num, seconds)
        histo.Fill(seconds)
        total_time += seconds
    # A line which represents the average is drawn in the TGraph
    avg = histo.GetMean()
    avg_line = ROOT.TLine(1,avg,last_event, avg)
    avg_line.SetLineColor(4)
    avg_line.SetLineWidth(2)
    # Draw and save!
    graph_canvas = ROOT.TCanvas('graph_canvas')
    graph_canvas.cd()
    graph.Draw("ALP")
    avg_line.Draw("Same")
   

    # Write graph to file
    graph_canvas.Print("%s/graph.png" %outdir,"png")
    graph.Write()
    graph_canvas.Write()    
    histo_canvas = ROOT.TCanvas('histo_canvas')
    histo_canvas.cd()
    histo.Draw('')

    # Write histogram to file
    histo_canvas.Print("%s/histo.png" %outdir,"png")
    histo.Write()
    histo_canvas.Write() 
    
    myfile.Close()   
    
    ########################                
    #### The html page! ####
    ########################
    titlestring = '<b>Report executed with release %s on %s.</b>\n<br>\n<hr>\n'\
                                   %(os.environ['CMSSW_VERSION'], time.asctime())
    html_file_name = '%s/%s_TimingReport.html' %(outdir, logfile_name[:-4])
    html_file = open(html_file_name,'w')
    html_file.write('<html>\n<body>\n'+\
                    titlestring)
    html_file.write('<table>\n'+\
                    '<tr>\n<td><img  src=graph.png></img></td>\n'+\
                    '<td><img  src=histo.png></img></td>\n</tr>\n'+\
                    '</table>\n')
    html_file.write('<hr>\n<h2>Time Report</h2>\n<pre>\n' + report + '</pre>\n')
    html_file.write('</body>\n</html>')
    html_file.close()

    ##########################
    #### Print statistics ####
    ##########################
    total_events = max(data)
    average_time = total_time / total_events
    sum = 0.
    for i in range(1, max(data)+1):
        sum += (data[i]-average_time)**2
    denominator = total_events**2 - total_events
    uncertainty = sqrt(sum/denominator)
    # Comment out next 2 line to round uncertainty to the most significant digit
    #rounded_uncertainty=round(uncertainty, -int(floor(log10(uncertainty))))
    #print 'Rounded uncertainty=' , rounded_uncertainty  
    print '------ Statistics ------'
    print 'last event = {}'.format(last_event)
    print 'Minval = {} maxval = {} interval = {}'.format(min_val, max_val, interval)
    print 'Total Time = {}'.format(total_time)
    print 'Average Time = {}'.format(average_time)
    print 'Uncertainty of Average Time = {} +/- {}'.format(average_time, uncertainty)

#################################################################################################    
        
if __name__ == '__main__':

    # Here we define an option parser to handle commandline options..
    usage = 'timing_parser.py <options>'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in_  profile',
                      help='The profile to manipulate' ,
                      default='',
                      dest='profile')
                      
    parser.add_option('-o', '--outdir',
                      help='The directory of the output' ,
                      default='',
                      dest='outdir')

    parser.add_option('-n', 
                      help='Number of secs per bin. Default is 1.' ,
                      default='1',
                      dest='startevt')                      
    (options,args) = parser.parse_args()
    
    # Now some fault control..If an error is found we raise an exception
    if options.profile == '' or\
       options.outdir == '':
        raise Exception('Please select a profile and an output dir!')
    
    if not os.path.exists(options.profile) or\
       not os.path.exists(options.outdir):
        raise Exception('Outdir or input profile not present!')
    
    try:
        startevt = float(options.startevt)        
    except ValueError:
         print 'Problems in convertng starting event value!'

    # launch the function!
    manipulate_log(options.outdir,options.profile,startevt)
