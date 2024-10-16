import json
import glob
from tqdm import tqdm

from openai import OpenAI
client = OpenAI()

#client.batches.cancel('batch_PquqjXRXbkCQ0IUpWdUofXtm')
# batch_job = client.batches.retrieve('batch_PquqjXRXbkCQ0IUpWdUofXtm')
# print(batch_job)
file_name = "batch_tasks.jsonl"
batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

# batch_job = client.batches.retrieve(batch_job.id)
# print(batch_job)
# Batch(id='batch_PquqjXRXbkCQ0IUpWdUofXtm', completion_window='24h', created_at=1721021992, endpoint='/v1/chat/completions', input_file_id='file-3nzX5lIaXq8B3yjAsH0YMeIq', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1721108392, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))