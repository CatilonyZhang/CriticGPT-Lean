import pathlib
import shutil

from tqdm import tqdm

input_dir = pathlib.Path("/home/jia/auto_proofs_v2_save")
output_dir = pathlib.Path("/home/jia/auto_proofs_v2")

i = 0
for file in tqdm(input_dir.glob("*.json")):
    # file name = e134a1aa-f210-5b06-82c7-68f2321d2a0f_e134a1aa-f210-5b06-82c7-68f2321d2a0f_5_542.json
    uuid = file.name.split("_")[0]
    proof_id = "_".join(file.name.split("_")[1:])
    uuid_dir = output_dir / uuid
    uuid_dir.mkdir(exist_ok=True, parents=True)

    # copy file to uuid_dir
    shutil.copy(file, uuid_dir / proof_id)

    # i += 1
    # if i > 1000:
    #     break
