import os
import pickle

print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Looking for credentials.json at: {os.path.join(os.getcwd(), 'credentials.json')}")
print(f"credentials.json exists: {os.path.exists('credentials.json')}")
print(f"token.pickle exists: {os.path.exists('token.pickle')}")
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

print("==== Google Drive Authentication Test ====")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check for credentials file
creds_path = 'credentials.json'
print(f"credentials.json exists: {os.path.exists(creds_path)}")

# Check for token file
token_path = 'token.pickle'
print(f"token.pickle exists: {os.path.exists(token_path)}")

# Try to authenticate
creds = None
SCOPES = ['https://www.googleapis.com/auth/drive']

if os.path.exists(token_path):
    with open(token_path, 'rb') as token:
        print("Loading token.pickle...")
        creds = pickle.load(token)
else:
    print("token.pickle not found")

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        print("Refreshing expired credentials...")
        creds.refresh(Request())
    else:
        if os.path.exists(creds_path):
            print("Starting new authorization flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(token_path, 'wb') as token:
                print(f"Saving new token to {token_path}")
                pickle.dump(creds, token)
        else:
            print("ERROR: credentials.json file not found!")

if creds:
    print("Successfully authenticated!")
    service = build('drive', 'v3', credentials=creds)
    print("Successfully created Drive service!")
else:
    print("Failed to authenticate!")

print("==== Test Complete ====")