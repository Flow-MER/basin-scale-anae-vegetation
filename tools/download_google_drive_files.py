"""
Download all files from Google Drive folder using Python

Usage: python download_drive_files.py

First run will open browser for authentication.
Credentials saved to token.json for future runs.
"""

import os
import io
import ee
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()
# === Determine script and project directories ===
SCRIPT_DIR = Path(__file__).resolve().parent  # Tools/
PROJECT_DIR = SCRIPT_DIR.parent                # Parent folder (where project.py lives)
OUTPUT_DIR = PROJECT_DIR / 'output/ndvi/avhrr_modis'
GEE_PROJECT = os.getenv('GEE_PROJECT', None)
GOOGLE_DRIVE_FOLDER_NAME = 'GEE_AVHRR_MODIS_NDVI'

def authenticate():
    """Authenticate with Google Drive API using GEE credentials."""
    # Initialize GEE to get authenticated credentials
    try:
        ee.Initialize(project=GEE_PROJECT)
    except:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    
    # Get credentials from GEE session
    credentials = ee.data.get_persistent_credentials()
    
    # Build Drive service with same credentials
    return build('drive', 'v3', credentials=credentials)

def find_folder(service, folder_name):
    """Find folder ID by name."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    items = results.get('files', [])
    
    if not items:
        print(f"❌ Folder '{folder_name}' not found")
        return None
    
    if len(items) > 1:
        print(f"⚠️  Multiple folders named '{folder_name}' found, using first one")
    
    return items[0]['id']

def list_files_in_folder(service, folder_id):
    """List all files in folder."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields='files(id, name, size, modifiedTime)',
        pageSize=1000
    ).execute()
    
    return results.get('files', [])

def download_file(service, file_id, file_name, output_path):
    """Download a single file."""
    request = service.files().get_media(fileId=file_id)
    
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
    
    # Write to disk
    with open(output_path, 'wb') as f:
        f.write(fh.getvalue())

def download_all_files(folder_name=GOOGLE_DRIVE_FOLDER_NAME, output_dir=OUTPUT_DIR):
    """Download all files from Drive folder."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Authenticate
    print("Authenticating with Google Drive...")
    service = authenticate()
    print("✓ Authenticated")

    # Find folder
    print(f"\nSearching for folder: {folder_name}")
    folder_id = find_folder(service, folder_name)
    if not folder_id:
        return
    print(f"✓ Found folder (ID: {folder_id})")

    # List files
    print("\nListing files...")
    files = list_files_in_folder(service, folder_id)
    print(f"✓ Found {len(files)} files")

    if not files:
        print("No files to download")
        return

    # Check which files already exist (check for both CSV and ZIP)
    existing_files = set(os.listdir(output_dir))
    existing_zips = {
        f.replace(".zip", ".csv") for f in existing_files if f.endswith(".zip")
    }
    files_to_download = [
        f
        for f in files
        if f["name"] not in existing_files and f["name"] not in existing_zips
    ]

    if not files_to_download:
        print("\n✓ All files already downloaded")
    else:
        print(f"\nDownloading {len(files_to_download)} new files...")
        print(f"(Skipping {len(files) - len(files_to_download)} existing files)")

        # Download and zip files
        for i, file in enumerate(files_to_download, 1):
            if not file["name"].endswith(".csv"):
                continue

            output_path = os.path.join(output_dir, file["name"])
            zip_path = os.path.join(output_dir, file["name"].replace(".csv", ".zip"))
            size_mb = int(file.get("size", 0)) / 1024 / 1024

            print(
                f"[{i}/{len(files_to_download)}] {file['name']} ({size_mb:.1f} MB)...",
                end=" ",
            )

            try:
                download_file(service, file["id"], file["name"], output_path)

                # Zip the CSV and remove original
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(output_path, file["name"])
                os.remove(output_path)

                print("✓")
            except Exception as e:
                print(f"❌ Error: {e}")

        print(f"\n✓ Download complete: {output_dir}")

    # Zip any remaining CSV files
    zip_remaining_csvs(output_dir)

    print(f"Total files: {len(os.listdir(output_dir))}")


def zip_remaining_csvs(directory):
    """Zip any remaining CSV files individually."""
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    if not csv_files:
        return

    print(f"\nZipping {len(csv_files)} remaining CSV files...")

    for csv_file in csv_files:
        csv_path = os.path.join(directory, csv_file)
        zip_path = os.path.join(directory, csv_file.replace(".csv", ".zip"))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, csv_file)
        os.remove(csv_path)

    print(f"✓ Zipped {len(csv_files)} CSV files individually")
    print(f"✓ Removed {len(csv_files)} original CSV files")


if __name__ == "__main__":
    download_all_files()
