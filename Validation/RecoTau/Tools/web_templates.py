main_page_template = '''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
  <link type="text/css" href="http://cern.ch/cms-project-tau-validation/styles.css" rel="stylesheet">
  <title>Tau ID Validation</title>
</head>
<body class="index">
  <div id="header">
    <A href="https://cms-project-tauvalidation.web.cern.ch/cms-project-tauvalidation/">Home</A>
    <A href="https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideTauValidation">Wiki</A>
  </div>
  <hr>
  <div id="title">
    <h1>Tau ID Validation</h1>
  </div>
  <hr>
  <h2>Official releases validated:</h2>
  <ul>
%s
  </ul>
  <h2>Special RelVal Validation:</h2>
  <ul>
%s
  </ul>
  <h2>Custom validation, tests and other stuff:</h2>
  <ul>
%s
  </ul>
</body>
</html>
'''
create_main_list_element = lambda x: '    <li><a href="%s/">%s</a></li>\n' % (x,x)

from string import Template
usual_validation_template = Template('''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
  <link type="text/css" href="http://cern.ch/cms-project-tau-validation/styles.css" rel="stylesheet">
  <title>$THIS_RELEASE Validation Results</title>
</head>
<body class="index">
  <div id="header">
    <A href="https://cms-project-tauvalidation.web.cern.ch/cms-project-tauvalidation/">Home</A>
    <A href="https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideTauValidation">Wiki</A>
  </div>
  <hr>
  <div id="title">
    <h1>$THIS_RELEASE Validation Results</h1>
  </div>
  <hr>
  <h2>Configuration:</h2>
  <pre>
$CONFIG
  </pre>
  <h2>Datasets:</h2>
$DATASETS
</body>
</html>
''')

def usual_validation_dataset_template(dataset_name, root_link, ref_file, dir_link, paths_to_pictures, source_file):
    chunk_list     = lambda l,n: [l[i:i+n] for i in range(0, len(l), n)]
    rows_paths     = chunk_list(paths_to_pictures,2)
    cell_template  = '        <td style="width: 640px;"><A href=%s><IMG src="%s" width="640" align="center" border="0"></A></td>'
    row_template   = '''      <tr>
%s      </tr>\n'''
    rows           = ''.join([row_template % ''.join([ cell_template % (path,path) for path in row]) for row in rows_paths])
    reference      = '<A href=%s>Reference</A>' % ref_file if ref_file else 'Reference not available'
    source         = '<A href=%s>Input Source</A>'    % source_file if ref_file else 'Source not available' #ALL source files seem empty... need to check
    return '''  <h3>%s ( <A href=%s>Root File</A> ) ( %s ) ( %s ) ( <A href=%s>Full Directory Content</A> )</h3>
  <table style="text-align: left; " border="1" cellpadding="2" cellspacing="1">
    <tbody>
%s
    </tbody>
  </table>
''' % (dataset_name, root_link, reference, source, dir_link, rows)
