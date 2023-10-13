ARG BASE_IMAGE=python:3.10-bullseye
FROM ${BASE_IMAGE}
USER root

RUN apt update && apt install --no-install-recommends -y \
    git curl make ssh openssh-client bash zip unzip htop vim

RUN python3 -m pip install --upgrade pip setuptools requests \
    && python3 -m pip install poetry

RUN ls 

# COPY ../pyproject.toml ../poetry.lock poetry.toml .
RUN poetry install || echo "skip to run poetry install."