import json
import os
import shutil
import requests


class Zenodo:
    def __init__(self, datasets):
        self.root_path = "./results"
        self.datasets = datasets

        self.zenodo_metadata = self.get_zenodo_metadata()
        self.baseurl = "https://zenodo.org/api"
        self.headers = {"Content-Type": "application/json"}
        access_token = os.environ.get('access_token')
        if access_token is None:
            print("Access token not found")
        else:
            self.params = {'access_token': access_token }

    @staticmethod
    def get_zenodo_metadata():
        metadata_zenodo_path = "metadata-zenodo.json"
        with open(metadata_zenodo_path, 'r') as f:
            return json.loads(f.read())

    def delete_all(self):
        ret = self.get_user_depositions()
        for dep in ret:
            print(dep["id"])
            self.delete(dep["id"])


    def delete(self, deposition_id):
        r = requests.delete(f'{self.baseurl}/deposit/depositions/{deposition_id}', headers=self.headers, params=self.params)
        self.handle_return(r)

    def upload(self):
        print("Write the deposition id of a dataset to to update it, or enter nothing to create a new deposition")
        a = input()
        if a == "":
            ret_json = self.new(self.zenodo_metadata)
        else:
            ret_json = self.get_user_depositions(a)

        self.upload_datasets(self.datasets, ret_json)

        print("Do you want to make the deposition permanently public (publish)? Write 'Yes I am sure'. You can always publish later from the website")
        a = input()
        if a.lower() == "yes i am sure":
            self.publish()
        else:
            return self.exit_upload(ret_json)

    def exit_upload(self, ret_json):
        link = ret_json["links"]["html"]
        print(f"Link to your artifact:\n{link}")

    def new(self, metadata):
        ret = self.post_new_deposition(metadata)
        print("Created a new deposition: {ret.json()['id']}")
        return ret.json()

    def post_new_deposition(self, metadata):
        ret = requests.post(f"{self.baseurl}/deposit/depositions", headers=self.headers, params=self.params, data=json.dumps(metadata))
        self.handle_return(ret)
        return ret

    def get_user_depositions(self, deposition_id=None):
        if deposition_id is None:
            ret = requests.get('https://zenodo.org/api/deposit/depositions', params=self.params)
            self.handle_return(ret)
        else:
            ret = requests.get(f'https://zenodo.org/api/deposit/depositions/{deposition_id}', params=self.params)
            self.handle_return(ret)

        return ret.json()

    def zip_folder(self, path):
        shutil.make_archive(path, 'zip', path)

    def zip_and_remove_folder(self, path):
        self.zip_folder(path)
        shutil.rmtree(path)

    def unpack_and_remove_zip(self, path):
        shutil.unpack_archive(f'{path}.zip', path)
        os.remove(f"{path}.zip")

    def handle_return(self, ret):
        if ret.status_code//100 != 2:
            print(ret.status_code, ret.json)


    def upload_datasets(self, datasets, ret_json):
        bucket_url = ret_json["links"]["bucket"]
        print(bucket_url)

        for dataset in datasets:
            dataset_path = f"{self.root_path}/{dataset}"

            results_path = f"{dataset_path}/results"
            self.zip_and_remove_folder(results_path)

            source_path = f"{dataset_path}/source"
            self.zip_and_remove_folder(source_path)

            # Zipping up dataset
            self.zip_folder(dataset_path)

            # Uploading datset
            with open(f"{dataset_path}.zip", "rb") as f:
                ret = requests.put(f"{bucket_url}/{dataset}.zip", data=f, params=self.params)
            self.handle_return(ret)

            # Upload was successful
            self.unpack_and_remove_zip(results_path)
            self.unpack_and_remove_zip(source_path)
            os.remove(f"{dataset_path}.zip")

        print("Upload succesful")

    def publish(self):
        deposition_id = self.id
        ret = requests.post(f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish', params=self.params )
        self.handle_return(ret)
        print("Successfully published")
        return ret

