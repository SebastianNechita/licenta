from concurrent.futures import ThreadPoolExecutor

from pedect.utils.constants import MAX_WORKERS

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
print("Created ThreadPoolExecutor with " + str(MAX_WORKERS) + " threads!")