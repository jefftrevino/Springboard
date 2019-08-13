# event posting code for Monterey Airport Weather Almanacs
# Jeff Trevino, 2019
# imports
from __future__ import print_function
import datetime
import pickle
import os.path

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar']

risk_ratios = pd.read_csv('calendar_obscuration_risk.csv', index_col=0, names=['risk_ratio'])
risk_ratios.index = risk_ratios.index + 1

predictions = pd.read_csv('predictions.csv', index_col=0, names=['temp', 'hum', 'obsc'], header=0, parse_dates=True)
mask = (predictions.index.date == 29) & (predictions.index.month == 2) # remove leap years
predictions = predictions[~mask]

# create day of year column
predictions['dayofyear'] = predictions.index.dayofyearx = predictions
x['dayofyear'] = x['dayofyear'].apply(lambda day: day - 1 if day > 60 else day)

# a helper function looks up the risk ratio by the day of the year
def get_risk_ratio_by_dayofyear(row):
    return risk_ratios.loc[row['dayofyear']]
# add risk_ratio to prediction frame
predictions['obscuration_risk_ratio'] = predictions.apply(get_risk_ratio_by_dayofyear, axis=1)
# drop dayof year
predictions = predictions.drop(columns=['dayofyear'])
x = predictions

# calendar setup
creds = None
# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

         # If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('calendar', 'v3', credentials=creds) # uncomment to sign in

# get a calendar list
page_token = None
while True:
  calendar_list = service.calendarList().list(pageToken=page_token).execute()
  for calendar_list_entry in calendar_list['items']:
    print(calendar_list_entry['accessRole'], ":", calendar_list_entry['summary'])
  page_token = calendar_list.get('nextPageToken')
  if not page_token:
    break

    # get calendar id
el_cid = calendar_list['items'][4]['id']

def build_daily_string(row):
    the_string = ''
    the_string += 'temp: ' + '{:.0f}'.format(row['temp']) + ' F ' + '\n'
    the_string += 'hum: ' + '{:.0f}'.format(row['hum']) + '%' + '\n'
    the_string += 'obsc(0-8): ' + '{:.0f}'.format(row['obsc']) + '\n'
    the_string += 'ORR: ' + '{:.2f}'.format(row['obscuration_risk_ratio']) + '\n'
    return the_string

def make_event_body(date_string, the_string):
    body = {'summary': the_string,
             'location': 'Monterey Airport',
             'description': 'A weather prediction for event planners',
             'start': {
                 'date': date_string
             },
             'end': {
                 'date': date_string
             },
            }
    return body

def make_event_metadata(row, eid, cid, service):
    # define a patch (the info to add in custom fields)
    body = {
      'extendedProperties': {
        'private': {
            'temperature': '{:.0f}'.format(row['temp']) + ' F ',
            'humidity': '{:.0f}'.format(row['hum']) + '%',
            'obscuration(0-8)': '{:.0f}'.format(row['obsc']),
            'obscuration risk ratio': '{:.2f}'.format(row['obscuration_risk_ratio'])
        }
      }
    }


def event_from_row(row, cid):
    date_string = str(index.date())
    event_summary_string = build_daily_string(row)
    mr_body = make_event_body(date_string, event_summary_string)
    event = service.events().insert(calendarId=cid, body=mr_body).execute()
#     print('Event created: %s' % (event.get('htmlLink'))) # maybe add metadata in future
#     so_meta = make_event_metadata(row, event['id'], cid, service)
#     service.events().patch(calendarId=cid, eventId=event['id'], body=so_meta).execute()

mask = (predictions.index.year >= 2019) & (predictions.index.month >= 9)
predictions = predictions[mask]
# get ahold of the first date
sept_first = predictions.loc['2019-09-01':'2019-09-01']
# build a prediction string for the data and post it to the calendar
for index, row in sept_first.iterrows():
    date_string = str(index.date())
    event_summary_string = build_daily_string(row)
    mr_body = make_event_body(date_string, event_summary_string)
    service.events().insert(calendarId=el_cid, body=mr_body).execute()
