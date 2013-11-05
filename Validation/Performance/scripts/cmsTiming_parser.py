#! /usr/bin/env python


def get_max(data,index=1):
    max_time=-1
    for el in data:
        sec=el[index]
        if max_time<sec:
            max_time=sec
    return max_time

def get_min(data,index=1):
    min_time=1e20
    for el in data:
        sec=el[index]
        if min_time>sec:
            min_time=sec
    return min_time    
    
def manipulate_log(outdir,logfile_name,secsperbin):

    import time
    import sys
    import ROOT       
    from math import sqrt, log10, floor

    # the fundamental structure: the key is the evt number the value is a list containing
    # VSIZE deltaVSIZE RSS deltaRSS
    data=[]
    report = '' 
    # open file and read it and fill the structure!
    logfile=open(logfile_name,'r')
    logfile_lines=logfile.readlines()
    logfile.close()

    # we get the info we need!
    i=0
    parse_report=True
    while i < len(logfile_lines):
        line=logfile_lines[i]
        if 'TimeEvent>' in line:
            line=line[:-1] #no \n!
            line_content_list=line.split(' ')
            event_number=int(line_content_list[1])
            seconds=float(line_content_list[3])
            data.append((event_number,seconds))
        elif parse_report and'TimeReport' in line:
            # add an empty line before report's tables
            if '[sec]' in line:
                report+='\n'
                report += line.replace('TimeReport', '')
            # remove two non-informative lines
            elif 'headings' in line:
               i+=1
               continue
            # parsing last summaries
            elif 'complete' in line:
               report += '\n'
               report += line.replace('TimeReport', '')
               for k in range(12):
                   report += logfile_lines[i]
                   i+=1
               parse_report=False
            # add simple report line
            else:
               report += line.replace('TimeReport', '') 
        i+=1

    # init Graph and histo
    
    # The Graphs 
    __argv=sys.argv # trick for a strange behaviour of the TApp..
    sys.argv=sys.argv[:1]
    ROOT.gROOT.SetStyle("Plain") # style paranoia
    sys.argv=__argv
    #Cannot use this option when the logfile includes
    #a large number of events... PyRoot seg-faults.
    #Set ROOT in batch mode to avoid canvases popping up!
    ROOT.gROOT.SetBatch(1)

    # Save in file
    rootfilename='%s/graphs.root' %outdir
    myfile=ROOT.TFile(rootfilename,'RECREATE')   
    
    
    # Set fancy limits
    min_val=get_min(data,1)
    max_val=get_max(data,1)
    interval=max_val-min_val
    
    min_val=min_val-interval*0.2
    max_val=max_val+interval*0.2
    interval=max_val-min_val
    
    nbins=int(interval/secsperbin)
    
    print 'Minval =', min_val,' maxval =',max_val, ' interval =',interval
    
    histo=ROOT.TH1F('Seconds per event','Seconds per event',nbins,min_val,max_val)
    histo.GetXaxis().SetTitle("s")    
    
    npoints=len(data)   
    
    graph=ROOT.TGraph(npoints)
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)
    graph.SetLineColor(2)        
    graph.SetTitle('Seconds per event')
    graph.SetName('SecondsPerEvent')    
    graph.GetXaxis().SetTitle("Event")
    
    last_event=data[-1][0]
    print 'last event =',last_event
    graph.GetXaxis().SetLimits(0,last_event)
        
    graph.GetYaxis().SetTitleOffset(1.3)
    graph.GetYaxis().SetTitle("s")
    graph.GetYaxis().SetRangeUser(0,max_val)

    
    
    # Fill them
    
    evt_counter=0
    total=0
    for evt_num,secs in data:
        graph.SetPoint(evt_counter,evt_num,secs)
        histo.Fill(secs)
        total+=secs
        evt_counter+=1
        
    average=total/evt_counter
    
    sum=0.
    for i in range(evt_counter):
        sum+=(data[i][1]-average)**2
    uncertainty= sqrt(sum/(evt_counter*(evt_counter-1)))
 
    # round uncertainty to the most significant digit
    #rounded_uncertainty=round(uncertainty, -int(floor(log10(uncertainty))))
    #print 'Rounded uncertainty=' , rounded_uncertainty  

    print 'Total Time =', total
    print 'Average Time =', average    
    print 'Uncertainty of Average Time =', average, '+/-', uncertainty

    #A line which represents the average is drawn in the TGraph
    avg=histo.GetMean()
    avg_line=ROOT.TLine(1,avg,last_event,avg)
    avg_line.SetLineColor(4)
    avg_line.SetLineWidth(2)
        
    # draw and save!
    graph_canvas=ROOT.TCanvas('graph_canvas')
    graph_canvas.cd()
    graph.Draw("ALP")
    avg_line.Draw("Same")
    
    graph_canvas.Print("%s/graph.png" %outdir,"png")
    
    # write it on file
    graph.Write()
    graph_canvas.Write()    

    histo_canvas=ROOT.TCanvas('histo_canvas')
    histo_canvas.cd()
    histo.Draw('')

    histo_canvas.Print("%s/histo.png" %outdir,"png")
    
    # write it on file
    histo.Write()
    histo_canvas.Write() 
    
    myfile.Close()   
                    
    # The html page!------------------------------------------------------------------------------
    
    titlestring='<b>Report executed with release %s on %s.</b>\n<br>\n<hr>\n'\
                                   %(os.environ['CMSSW_VERSION'],time.asctime())
        
    html_file_name='%s/%s_TimingReport.html' %(outdir,logfile_name[:-4])# a way to say the string until its last but 4th char
    html_file=open(html_file_name,'w')
    html_file.write('<html>\n<body>\n'+\
                    titlestring)
    html_file.write('<table>\n'+\
                    '<tr>\n<td><img  src=graph.png></img></td>\n'+\
                    '<td><img  src=histo.png></img></td>\n</tr>\n'+\
                    '</table>\n')
    html_file.write('<hr>\n<h2>Time Report</h2>\n<pre>\n' + report + '</pre>\n')
    html_file.write('</body>\n</html>')
    html_file.close()    
    
    
#################################################################################################    
        
if __name__ == '__main__':
    
    import optparse
    import os
    
    # Here we define an option parser to handle commandline options..
    usage='timing_parser.py <options>'
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
    if options.profile=='' or\
       options.outdir=='':
        raise('Please select a profile and an output dir!')
    
    if not os.path.exists(options.profile) or\
       not os.path.exists(options.outdir):
        raise ('Outdir or input profile not present!')
    
    try:
        startevt=float(options.startevt)        
    except ValueError:
         print 'Problems in convertng starting event value!'
         
            
    #launch the function!
    manipulate_log(options.outdir,options.profile,startevt)
    
    
