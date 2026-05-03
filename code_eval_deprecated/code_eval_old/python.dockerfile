# Use the official Python image (Debian Bookworm-based)
FROM python:3.12-slim-bookworm
# Just work from root
# WORKDIR /
COPY runner_inside.py .
# NOTE: no real dependencies rn because these scripts should work
# without them.
# RUN touch requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "runner_inside.py"]