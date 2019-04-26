from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from pedect.utils.constants import MAX_WORKERS

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

mainMutex = Lock()

print("Created ThreadPoolExecutor with " + str(MAX_WORKERS) + " threads!")