from datasets import load_dataset


class DatasetManager:

    def __init__(self):
        self.base_datasets = {
            "math-train": load_dataset("AI-MO/math-base-v1.0", split="train"),
            "math-test-formal": load_dataset("AI-MO/math-base-v1.0", split="test"),
            "aops-formal": load_dataset("AI-MO/aops-base-v2.0", split="train"),
        }

        self._uuid_map = {}
        for key, dataset in self.base_datasets.items():
            for i, item in enumerate(dataset):
                self._uuid_map[item["uuid"]] = (key, i)

        print(f"Number of samples loaded: {len(self._uuid_map)}")

    def get_problem_by_uuid(self, uuid):
        key, i = self._uuid_map[uuid]
        return self.base_datasets[key][i]

    def build_formal_proof_dataset(self):
        pass


if __name__ == "__main__":
    manager = DatasetManager()
    uuid = "359724f8-2666-5ef7-a9fa-2491e8038388"
    print(manager.get_problem_by_uuid(uuid))
