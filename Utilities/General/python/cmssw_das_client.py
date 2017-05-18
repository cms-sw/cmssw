from commands import getstatusoutput
from json import loads

#Copied from das_client.py
def get_value(data, filters, base=10):
  """Filter data from a row for given list of filters"""
  for ftr in filters:
    if  ftr.find('>') != -1 or ftr.find('<') != -1 or ftr.find('=') != -1:
      continue
    row = dict(data)
    values = []
    keys = ftr.split('.')
    for key in keys:
      val = [v for v in extract_value(row, key, base)]
      if  key == keys[-1]: # we collect all values at last key
        values += [json.dumps(i) for i in val]
      else:
        row = val
      if  len(values) == 1:
        yield values[0]
      else:
        yield values

def get_data(query, limit=None, threshold=None, idx=None, host=None):
  cmd_opts = "--format=json"
  if threshold is not None: cmd_opts += " --threshold=%s" % threshold
  if limit     is not None: cmd_opts += " --limit=%s"     % limit
  if idx       is not None: cmd_opts += " --idx=%s"       % idx
  if host      is not None: cmd_opts += " --host=%s"      % host
  err, out = getstatusoutput("das_client %s --query '%s'" % (cmd_opts, query))
  if not err: return loads(out)
  return {'status' : 'error', 'reason' : out}
