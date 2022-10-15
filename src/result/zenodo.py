import requests
import json
import os

class Zenodo:
    def __init__(self, dataset):
        self.path = dataset.path
        self.files = dataset.files

        self.metadata_zenodo_path = "metadata-zenodo.json"
        with open(self.metadata_zenodo_path, 'r') as f:
            self.zenodo_metadata = json.loads(f.read())

        self.baseurl = "https://zenodo.org/api"
        self.headers = {"Content-Type": "application/json"}
        access_token = os.environ.get('access_token')
        if access_token is None:
            print("Access token not found")
        else:
            self.params = {'access_token': access_token }


    def upload(self):
        print("Do you want to create a new deposition for the dataset?")
        a = input()
        if a.lower() == "yes" or a.lower() == "y":
            self.new(self.zenodo_metadata)
        else:
            return
        print("Do you want to upload the dataset files to the deposition?")
        a = input()
        if a.lower() == "yes" or a.lower() == "y":
            self.upload_files(self.files)
        else:
            return
        print("Do you want to make the deposition public (publish)? Write 'Yes I am sure'. You can always publish later from the website")
        a = input()
        if a.lower() == "yes i am sure":
            self.publish()



    def new(self, metadata):
        r = self.post_new_deposition(metadata)
        self.r_json = r.json()
        #print(self.r_json)
        self.id = self.r_json["id"]

    def post_new_deposition(self, metadata):
        # get request, returns our response
        r = requests.post(f"{self.baseurl}/deposit/depositions", headers=self.headers, params=self.params, data=json.dumps(metadata))

        # response status
        if r.status_code != 200:
            print(r.status_code)
            # response data, formatted nicely
            print(json.dumps(r.json(), indent=2))

        return r

    def get_user_depositions(self):
        r = requests.get('https://zenodo.org/api/deposit/depositions', params=self.params)
        if r.status_code != 200:
            print(r.status_code)
        return r.json()


    def upload_files(self, files):
        bucket_url = self.r_json["links"]["bucket"]
        # TODO: Add zipping of results
#        shutil.make_archive(output_filename, 'zip', dir_name)

        for filename in files:
            file_path = f"{self.path}/{filename}"
            with open(file_path, "rb") as f:
                r = requests.put(f"{bucket_url}/{filename}", data=f, params=self.params)
            if r.status_code != 200:
                print(r.status_code, r.json)

        return True

    def publish(self):
        deposition_id = self.id
        r = requests.post(f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish', params=self.params )
        if r.status_code != 200:
            print(r.status_code)
        else:
            print("Successfully published")

