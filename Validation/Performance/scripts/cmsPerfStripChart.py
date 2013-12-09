#! /usr/bin/env python

import os, sys
try: import simplejson as json
except ImportError: import json

# Helper functions
def get_yaxis_range(list):
    """
    Given a list of dictionaries, where each dict holds the information
    about an IB, this function returns a tuple (low, high) with the lowest
    and the highest value of y-axis, respectively. 
    """
    low, high = sys.maxint, -1
    for node in list:
        low = min((node['average'] - node['error']), low)
        high = max((node['average'] + node['error']), high)
    return (low, high)

# Main operation function
def operate(timelog, memlog, json_f, num):
    """
    Main operation of the script (i.e. json db update, histograms' creation)
    with respect to the specifications of all the files & formats concerned.
    """
    import re
    import commands
    import ROOT
    from datetime import datetime

    script_name=os.path.basename(__file__)

    # Open files and store the lines.
    timefile=open(timelog, 'r')
    timelog_lines=timefile.readlines()
    timefile.close()

    memfile=open(memlog, 'r')
    memlog_lines=memfile.readlines()
    memfile.close()
 
    # Get average, uncertainty of average and maximum rss.
    max_rss=average=error=' '
    i=0
    while i<len(timelog_lines):
        line=timelog_lines[i]
        if 'Uncertainty of Average Time' in line:
            line=line[:-1]
            line_list=line.split(' ')
            average=line_list[5]
            error=line_list[7]
        i+=1
    i=0
    while i<len(memlog_lines):
        line=memlog_lines[i]
        if 'Maximum rss' in line:
            line=line[:-1]
            line_list=line.split(' ')
            max_rss=line_list[3]
            break
        i+=1

    # Get Integration Build's identifier
    IB=os.path.basename(commands.getoutput("echo $CMSSW_BASE"))

    # Check if log files were parsed properly...
    # and if the IB is valid using regular expressions.
    try: 
        # regex for a float
        regex="^\d+\.?\d*$"
        if average == ' ' or re.match(regex, average) is None:
            raise RuntimeError('Could not parse \"' + timelog + '\" properly. ' +\
                               'Check if Average Time is defined correctly.')
        if error == ' ' or re.match(regex, error) is None:
            raise RuntimeError('Could not parse \"' + timelog + '\" properly. ' +\
                               'Check if Uncertainty of Average Time is defined correctly.')
        if max_rss == ' ' or re.match(regex, max_rss) is None:
            raise RuntimeError('Could not parse \"' + memlog + '\" properly. ' +\
                               ' Check if Maximum rss is defined correct.')
 
       # regex for dates 'YYYY-MM-DD-HHMM'
        regex = '(19|20|21)\d\d-(0[1-9]|1[012])-(0[1-9]|[12]'+\
                '[0-9]|3[01])-([01][0-9]|2[0-4])([0-5][0-9])$'
        if re.search(regex, IB) is None:
            raise RuntimeError('Not a valid IB. Valid IB: ' +\
                               '[CMSSW_X_X_X_YYYY-MM-DD-HHMM]')
    except Exception, err:
        sys.stderr.write(script_name + ': Error: ' + str(err) + '\n')
        return 2
    
    # Open for getting the data.
    json_db=open(json_f, "r")
    dict=json.load(json_db)
    json_db.close()

    # Get the data to be stored and check if already exists.
    ib_list=IB.split('_')
    cmsrelease=ib_list[0] + '_' + ib_list[1] +\
               '_' + ib_list[2] + '_' + ib_list[3]
    data={"IB" : ib_list[4], "average" : float(average), "error" : float(error), "max_rss" : float(max_rss)}

    if data in dict["strips"]:
        sys.stderr.write(script_name + ": Warning: Entry already exists " +\
                                       "in json file and will not be stored! " +\
                                       "Only the strip charts will be created.\n")
    else:
        dict["strips"].append(data)
        print 'Storing entry to \"' + json_f +\
              '\" file with attribute values:\n' +\
              'IB=' + IB + '\naverage=' + average +\
              '\nUncertainty of average=' + error +'\nmax_rss=' + max_rss
        # Store the data in json file.
        json_db = open(json_f, "w+")
        json.dump(dict, json_db, indent=2)
        json_db.close()
        print 'File "' + json_f + '" was updated successfully!'

    # Change to datetime type (helpful for sorting).
    for record in dict["strips"]:
        time_list = record['IB'].split('-')
        d = datetime(int(time_list[0]), int(time_list[1]),  
                     int(time_list[2]), int(time_list[3][0:2]), 
                     int(time_list[3][2:]))
        record['IB'] = d

    # Sort the list.
    list = sorted(dict["strips"], key=lambda k : k['IB'], reverse=True)
 
    # Check if there are NUM entries.
    if num > len(list):
        new_num = len(list)
        sys.stderr.write(script_name + ': Warning: There are less than ' +\
                         str(num) + ' entries in json file. Changed number to ' +\
                         str(new_num) + '.\n')
        num = new_num

    # The histograms.
    ROOT.gROOT.SetStyle("Plain")
    outdir='.'

    # Save in file
    rootfilename=outdir + '/histograms.root'
    myfile=ROOT.TFile(rootfilename, 'RECREATE')  

    # Average time histogram.
    histo1=ROOT.TH1F("AveCPU per IB", "Ave CPU per IB", num, 0., num)
    histo1.SetTitle(cmsrelease + ": Showing last " + str(num) + " IBs")
    histo1.SetName('avecpu_histo')

    # Maximum rss histogram.
    histo2=ROOT.TH1F("Max rrs per IB", "Max rss per IB", num, 0., num)
    histo2.SetTitle(cmsrelease + ": Showing last " + str(num) + " IBs")
    histo2.SetName('maxrss_histo')

    # Fill in the histograms
    for i in range(num): 
        datime = list[i]['IB'].__format__('%Y-%b-%d %H:%M')
        average = list[i]['average']
        max_rss = list[i]['max_rss']
        error = list[i]['error']
      
        histo1.GetXaxis().SetBinLabel(num-i, datime)
        histo1.SetBinContent(num-i, average)   
        histo1.SetBinError(num-i, error)
        histo2.GetXaxis().SetBinLabel(num-i, datime)
        histo2.SetBinContent(num-i, max_rss)

    histo1.SetStats(0)   
    histo1.GetYaxis().SetTitle("Average CPU time")
    histo1.GetYaxis().SetTitleOffset(1.8)
    histo1.GetXaxis().SetTitle("Integration Build")  
    histo1.GetXaxis().SetTitleOffset(4.) 
    histo1.GetXaxis().CenterTitle()
    histo1.GetXaxis().LabelsOption('v')    
    # Histo1 - Set limits on the Y-axis
    min, max = get_yaxis_range(list)
    interval = max - min
    # ...get a bit more space
    min = min-interval*0.1
    max = max+interval*0.1
    histo1.GetYaxis().SetRangeUser(min, max)

    histo2.SetStats(0)
    histo2.GetYaxis().SetTitle("Maximum rss")
    histo2.GetYaxis().SetTitleOffset(1.8)
    histo2.GetXaxis().SetTitle("Integration Build")
    histo2.GetXaxis().SetTitleOffset(4.)
    histo2.GetXaxis().CenterTitle()
    histo2.GetXaxis().LabelsOption('v')

    # Draw and save!

    ave_canvas = ROOT.TCanvas(cmsrelease + '_average_canvas')
    ave_canvas.SetGridy()
    ave_canvas.SetBottomMargin(0.28)
    ave_canvas.SetLeftMargin(0.18)
    ave_canvas.cd()
    # Histo1 - draw line
    histo1.SetLineColor(2)
    histo1.SetLineWidth(2)
    histo1.DrawCopy("HISTO L")
    # Histo1 - draw errors and markers
    histo1.SetLineColor(1)
    histo1.SetLineStyle(2)
    histo1.SetLineWidth(1)
    histo1.SetMarkerStyle(8)
    histo1.SetMarkerSize(.6)
    histo1.SetMarkerColor(1)
    histo1.Draw("E1P SAME")
    ROOT.gStyle.SetErrorX(0)
    ave_canvas.Print(outdir + "/average_cpu_histo.png","png")

    rss_canvas = ROOT.TCanvas(cmsrelease + '_maxrss_canvas')
    rss_canvas.SetGridy()
    rss_canvas.SetBottomMargin(0.28)
    rss_canvas.SetLeftMargin(0.18)
    rss_canvas.cd()
    # Histo2 - draw line
    histo2.SetLineColor(2)
    histo2.SetLineWidth(2)
    histo2.DrawCopy("L")
    # Histo2 - draw markers    
    histo2.SetMarkerStyle(8)
    histo2.SetMarkerSize(.6)
    histo2.SetMarkerColor(1)
    histo2.Draw("P SAME")
    rss_canvas.Print(outdir + "/maximum_rss_histo.png","png")

    # write them on file
    histo1.Write()
    ave_canvas.Write()
    histo2.Write()
    rss_canvas.Write()

  
########################################################################################### 

if __name__ == '__main__':

    import optparse, stat
 
    ################################
    # Definition of command usage. #
    ################################
    script_name= os.path.basename(__file__)
    usage = script_name + ' <options> -t TIMELOG -m MEMLOG'
    parser = optparse.OptionParser(usage)
    parser.add_option('-t', '--timelog',
                      action='store',
                      dest='timelog',
                      default='',
                      metavar='TIMELOG',
                      help='input file TIMELOG, the output of cmsTiming_parser.py')
    parser.add_option('-m', '--memlog',
                      action='store',
                      dest='memlog',
                      default='',
                      metavar='MEMLOG', 
                      help='input file MEMLOG, the output of cmsSimplememchecker_parser.py')
    parser.add_option('-j', '--jsonfile',
                     action='store',
                     dest='json_f',
                     default='strips.json',
                     metavar='FILE.JSON',
                     help='the .json file database')
    parser.add_option('-n', type='int',
                      action='store',
                      dest='num',
                      default='30',
                      metavar='NUM',
                      help='last NUM entries to be printed in the strip charts. Default is 30.')
    (options, args) = parser.parse_args()

    ######################################
    # Some error handling for the usage. #
    ######################################
    if options.timelog == '' or\
       options.memlog == '':
        sys.exit('%s: Missing file operands!\n' % script_name+\
                 'Type %s --help for more information!' % script_name)
    if not os.path.exists(options.timelog) or\
       not os.path.exists(options.memlog):
        sys.exit('%s: Error: Not present file(s)!' % script_name)

    #############################################
    # Validity of .json file-database.          #
    #############################################
 
    # The format that the json file must have:  
    format = "\n  {  \"strips\" :\n"   +\
             "    [\n      {\"IB\" : \"XXX_XXX\", \"average\" : M, \"error\" : E \"max_rss\" : N},\n" +\
             "        .........................................\n" +\
             "    ]\n"+\
             "  }\n"

    # json file validity checks start under the try statement
    json_db = open(options.json_f, "r+")
    try:
        # -check if the json file is empty; if yes, create a new database upon it
        if os.stat(options.json_f)[stat.ST_SIZE] == 0:
            sys.stderr.write(script_name + ': Warning: File \"' + options.json_f +\
                             '\" is empty. A new database will be created upon it.\n')
            json_db.write("{\n  \"strips\" : [\n  ]\n}\n")
            json_db.seek(0, 0)

        # -check if file loads as a valid json
        dict = json.load(json_db)

        # -check if strips key is there.(Look format above!)
        dict["strips"]

        # -check if value of strips is type of list
        if not isinstance(dict["strips"], list):
            raise Exception

        # -check if the list has valid elements
        if dict["strips"]:
            for item in dict["strips"]:
                if not set(['IB', 'average', 'error', 'max_rss']).issubset(item):
                    raise KeyError
    except ValueError: 
        sys.exit(script_name + ': Error: Not a valid json file! Please, check the format:\n' + format)
    except KeyError:
        sys.exit(script_name + ': Error: Invalid format in the json file! Check it here:\n' + format)
    finally:
        json_db.close()

    ####################
    # Start operation. #
    ####################

    # sys.exit() used in order to return an exit code to shell, in case of error
    sys.exit(operate(options.timelog, options.memlog, options.json_f, options.num))
