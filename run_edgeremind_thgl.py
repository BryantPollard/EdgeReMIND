"""
Wrapper script to run EdgeReMIND with SSL verification disabled.
This patches SSL before importing TGM to handle server-side certificate issues.
"""
import ssl
import sys
import os
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

if __name__ == '__main__':
    script_path = os.path.join(os.path.dirname(__file__), 'examples', 'linkproppred', 'thgl', 'edgeremind.py')
    sys.argv[0] = script_path
    with open(script_path) as f:
        exec(f.read())
