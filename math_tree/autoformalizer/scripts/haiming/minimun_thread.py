import concurrent.futures
import time


def run(i):
    try:
        print(f"Running {i}")
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Exception: {e}")


def run_process(i):
    print("Running process")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            executor.map(run, range(200))
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=256) as executor:
            executor.map(run_process, range(256))
    except Exception as e:
        print(f"Exception: {e}")
